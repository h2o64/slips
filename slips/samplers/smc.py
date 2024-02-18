# Sequential Monte-Carlo sampler

# Libraries
import torch
from tqdm import trange
import math
from .mcmc import ula_mcmc, mala_mcmc

# Make the initial distribution details for the Gaussian case


def init_sample_gaussian(n_samples, sigma, dim, device): return sigma * torch.randn((n_samples, dim), device=device)


def init_log_prob_and_grad_gaussian(x, sigma): return (-0.5 * (torch.sum(torch.square(x),
                                                                         dim=-1) / torch.square(sigma)) - 0.5 * math.log(2. * math.pi) / x.shape[-1], -x)

# Sequential Monte Carlo algorithm


def smc_algorithm(n_particles, target_log_prob, target_log_prob_and_grad, betas, n_mcmc_steps, init_sample, init_log_prob_and_grad,
                  langevin_type='mala', init_step_size=1e-2, skip_ressampling=False, use_ais=False, verbose=False):
    """Sequential Monte Carlo algorithm

    Args:
        n_particles (int): Number of particles
        target_log_prob_and_grad (function : Log-likelihood of the target distribution
        target_log_prob_and_grad (function : Log-likelihood and gradient of the target distribution
        betas (torch.Tensor for shape (K,)): Scheduling of the inverse temperatures
        n_mcmc_steps (int): Number of MCMC steps
        init_sample (function): Sample the initial distribution (takes an int as argument)
        init_log_prob_and_grad (function): Log-likelihood and gradient of the initial distribution
        langevin_type (str): Type of MCMC algorithm (default is 'mala')
        init_step_size (float): Initial step size (default is 1e-2)
        use_ais (bool): Whether to turn SMC into AIS
        verbose (bool): Display a progress bar (default is False)

    Returns:
        samples (torch.Tensor of shape (n_particles, *data_shape)): Approximate samples according to the target distribution
    """

    # Define intermediate distributions
    def log_prob_and_grad_k(k, x):
        # Evaluate the initial distribution
        init_log_prob_, init_grad = init_log_prob_and_grad(x)
        # Evaluate the target distribution
        target_log_prob_, target_grad = target_log_prob_and_grad(x)
        # Compute the log_prob and grad
        log_prob = (1. - betas[k]) * init_log_prob_ + betas[k] * target_log_prob_
        grad = (1. - betas[k]) * init_grad + betas[k] * target_grad
        return log_prob, grad

    def log_prob_k(k, x):
        # Evaluate the initial distribution
        init_log_prob_ = init_log_prob_and_grad(x)[0]
        # Evaluate the target distribution
        target_log_prob_ = target_log_prob(x)
        # Compute the log_prob
        log_prob = (1. - betas[k]) * init_log_prob_ + betas[k] * target_log_prob_
        return log_prob

    def score_k(k, x): return log_prob_and_grad_k(k, x)[1]
    # Sample the initial distirbution
    x = init_sample(n_particles)
    # Make the initial step size
    step_size = init_step_size
    # Browse all the temperatures
    if verbose:
        r = trange(1, betas.shape[0])
    else:
        r = range(1, betas.shape[0])
    if use_ais:
        log_weights_ais = torch.zeros((n_particles,), device=x.device)
    for k in r:
        # Compute the weights
        log_weights = log_prob_k(k, x) - log_prob_k(k - 1, x)
        if use_ais:
            log_weights_ais += log_weights
        weights = torch.nn.functional.softmax(log_weights, dim=0)
        # Resample the particles
        if not use_ais:
            weights = torch.nn.functional.softmax(log_weights, dim=0)
            idx = torch.multinomial(weights, n_particles, replacement=True)
            x = x[idx]
        # Run the MCMC
        def cur_score(x): return score_k(k, x)
        def cur_log_prob_and_grad(x): return log_prob_and_grad_k(k, x)
        if langevin_type == 'ula':
            x = ula_mcmc(x.clone(), step_size, cur_score, n_mcmc_steps)
        else:
            x, step_size = mala_mcmc(x.clone(), step_size, cur_log_prob_and_grad,
                                     n_mcmc_steps, per_chain_step_size=True)
    if use_ais:
        return x, torch.nn.functional.softmax(log_weights_ais, dim=0)
    else:
        return x
