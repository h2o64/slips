# Implementation of the One-At-a-Time (OAT) algorithm from Saeed Saremi, Ji Won Park, & Francis Bach. (2023). Chain of Log-Concave Markov Chains.
# WARNING: Here I use the basic Langevin algorithm

# Libraries
import torch
from tqdm import trange
from .mcmc import ula_mcmc, underdamped_langevin_mcmc
from .alphas import AlphaMnM
import math


def jump(y_bar, sigma_eff, sigma_sq_score):
    """Perform the jump step of the algorithm (Eq (2.3))

        E[X | Y_{1:m} = y_{1:m}] = \bar{y}_{1:m} + m^{-1} \\sigma^2 \nabla p(\bar{y}_{1:m}; m^{-1} \\sigma^2)

    Args:
        y_bar (torch.Tensor of shape (batch_size, *data_shape)): Average of y_1, ..., y_m
        sigma_eff (float): Value of m^{-1} \\sigma^2
        sigma_sq_score (function): Score of the noising distribution at sigma times sigma squared

    Returns:
        x (torch.Tensor of shape (batch_size, *data_shape)): Denoised sample
    """

    return y_bar + sigma_sq_score(y_bar, sigma_eff)


def cond_score(y, running_mean_y, sigma, sigma_eff, m, sigma_sq_score):
    """Compute the conditional score of y_{m+1} | y_{1:m} (Eq (4.3))

    Args:
        y (torch.Tensor of shape (batch_size, *data_shape)): Sample from the current m step
        running_mean_y (torch.Tensor of shape (batch_size, *data_shape)): Average of y_1, ..., y_{m-1}
        sigma (float): Value of sigma
        sigma_eff (float): Value of sigma / sqrt(m)
        m (int): Current m
        sigma_sq_score (function): Score of the noising distribution at sigma times sigma squared

    Returns:
        score (torch.Tensor of shape (batch_size, *data_shape)): Value of the conditional score
    """

    # Update the running mean
    y_bar = running_mean_y + ((y - running_mean_y) / m)
    # Compute the score
    return (sigma_sq_score(y_bar, sigma_eff) + y_bar - y) / torch.square(sigma)


def make_init(shape, sigma, device):
    """Default initialization method for Langevin samples

        Y ~ U([-1,1]) + sigma N(0,I)

    Args:
        shape (tuple of int): Shape of the desired samples
        sigma (float): Value of sigma
        device (torch.device): Device to store the samples

    Returns:
        y (torch.Tensor of shape shape): Obtained samples
    """

    x = torch.rand(shape, device=device)
    x = 2. * x - 1.
    return x + torch.randn_like(x) * sigma

def oat_sampler(
        y1,
        sigma,
        M,
        n_mcmc_steps,
        sigma_sq_score,
        step_size=1e-2,
        friction=None,
        make_init=make_init,
        verbose=False,
        callbacks=None):
    """Run the One-At-a-Time (OAT) algorithm

    Args:
        y1 (torch.Tensor of shape (batch_size, *data_shape)): Initial sample for the sampling of p(y_1)
        sigma (float): Value of sigma
        M (int): Value of M
        n_mcmc_steps (int): Number of Langevin steps to sample the conditional distribution
        sigma_sq_score (function): Score of the noising distribution at sigma times sigma squared
        step_size (float): Step size for the Langevin algorithm (default is 1e-2)
        make_init (function): Function to initialize the Langevin chains. The default is to use the make_init
            function. If None, it will denoise the sample from step m-1 and noise it to initialize the sample
            of step m.
        verbose (bool): Whether to display a fancy progress bar (default is False)
        callbacks (list): List of functions to execute with intermediate samples (default is None)

    Returns:
        x (torch.Tensor of shape (batch_size, *data_shape)): Approximate sample from the target distribution
    """

    # Initialize the running mean
    running_mean_y = 0.0
    # Compute a nice lipschitz constant
    if friction is not None:
        lipschitz_cte = 1 / sigma
    # Make the MnM version of sigma_sq_score
    def sigma_sq_score_mnm(y, sigma): return torch.square(sigma) * sigma_sq_score(y, 1.0, sigma, AlphaMnM())
    # Walk : Run over all the noise levels
    y_init = y1.clone()
    if verbose:
        r = trange(1, M + 1)
    else:
        r = range(1, M + 1)
    for m in r:
        # Compute the effective scale
        sigma_eff = sigma / math.sqrt(m)
        # Define the conditional score at stake
        def cond_score_(y): return cond_score(y, running_mean_y, sigma, sigma_eff, m, sigma_sq_score_mnm)
        # Sample with Langevin
        if friction is not None:
            y_m = underdamped_langevin_mcmc(y_init, step_size, friction, lipschitz_cte, cond_score_, n_mcmc_steps)
        else:
            y_m = ula_mcmc(y_init, step_size, cond_score_, n_mcmc_steps)
        # Update the running mean
        running_mean_y = running_mean_y + (y_m - running_mean_y) / m
        # Execute callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback(y_m, running_mean_y, m)
        # Update y_init
        if make_init is not None:
            y_init = make_init(y_init.shape)
        else:
            y_init = jump(running_mean_y, sigma_eff, sigma_sq_score_mnm) + sigma * torch.randn_like(y_init)
    # Jump
    return jump(running_mean_y, sigma_eff, sigma_sq_score_mnm)
