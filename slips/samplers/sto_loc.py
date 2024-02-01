# Generalized stochastic localization algorithm

# Libraries
import math
import torch
from tqdm import trange
from .integrators import *
from .alphas import *
from .mcmc import ula_mcmc

def get_nabla_log_pt_from_mc_est(score_est, score_type, y, t, sigma, alpha):
    """Get \nabla \log p_t from any type of score estimator

    Args:
        score_est (function): Estimator of the score
        score_type (str): Type of score estimator
        y (torch.Tensor of shape (batch_size, *data_shape)): State to evaluate
        t (float): Current time
        sigma (torch.Tensor): Noise level
        alpha (alphas.Alpha): Alpha object

    Returns:
        nabla_log_p_t_y (torch.Tensor with the same shape as y): Value of the score at y
    """
    score_ = score_est(y, t, sigma, alpha)
    if score_type == 'nabla_log_p_t':
        return score_
    elif score_type == 'mc':
        return (alpha.alpha(t) * score_ - y) / (torch.square(sigma) * t)
    elif score_type == 'mc_grad':
        return score_ / alpha.alpha(t)
    elif score_type == 'mc_reparam':
        return score_ / (sigma * math.sqrt(t))
    elif score_type == 'mc_grad_reparam':
        return score_ / alpha.alpha(t)
    else:
        raise NotImplementedError('Score type {} not implemented.'.format(score_type))

def sto_loc_algorithm(alpha, y_init, K, T, sigma, score_est, score_type='nabla_log_p_t',
    epsilon=1e-5, epsilon_end=0.0, use_exponential_integrator=True,
    use_logarithmic_discretization=False, use_snr_discretization=True,
    verbose=False, callbacks=None):
    """Generalized stochastic localization algorithm

    Args:
        alpha (Alpha): Object containing the implementation details of alpha(t)
        y_init (torch.Tensor of shape (batch_size, *data_shape)): Initial integration point
        K (int): Number of discretization steps
        T (float): Final time (this variable is overwriten with alpha.T if available)
        sigma (float): Value of sigma
        score_est (float): Estimator of the score
        score_type (str): Type of score estimator (in 'nabla_log_p_t', 'mc', 'mc_grad', 'mc_reparam', 'mc_grad_reparam') (default is 'nabla_log_p_t')
        epsilon (str): Value of epsilon (default is 1e-5)
        epsilon_end (str): The stopping time will be T - epsilon_end (default is 0.0)
        use_exponential_integrator (bool): Whether to use the exponential integrator when available (default is True)
        use_logarithmic_discretization (bool): Whether to use logarithmic discrétization when available (default is False)
        use_snr_discretization (bool): Whether to use the SNR to define the time discretization (default is True)
        verbose (bool): Whether to display the sampling progress (default is False)
        callbacks (list of functions or None): Callbacks to call on (y_t, t) during sampling (default is None)

    Return:
        ret (torch.Tensor with the same shape as y_init): Samples from the algorithm
    """

    # Initialize y
    y = y_init.clone()
    # If the alpha is not in finite time, remove epsilon_end and align T
    if alpha.is_finite_time:
        T = alpha.T
    else:
        epsilon_end = 0.0
    # Compute time discretization
    if use_snr_discretization:
        log_g_sq = lambda t : 2. * math.log(alpha.g(t))
        log_g_sq_inv = lambda t : alpha.g_inv(math.exp(t / 2.))
        ts = make_time_discretization_from_snr(log_g_sq=log_g_sq, log_g_sq_inv=log_g_sq_inv, T=T, K=K,
            eps_start=epsilon, eps_end=epsilon_end).to(y.device)
    else:
        split_in_middle = alpha.is_finite_time
        log_start = use_logarithmic_discretization
        log_end = use_logarithmic_discretization and alpha.is_finite_time
        ts = make_time_discretization(T=T, K=K, eps_start=epsilon, eps_end=epsilon_end,
            split_in_middle=split_in_middle, log_start=log_start, log_end=log_end).to(y.device)
    # Select the SDE solver
    if score_type == 'nabla_log_p_t':
        sde_solver_step = integration_step_nabla_log_p_t
    elif score_type == 'mc':
        sde_solver_step = integration_step_mc_est
    elif score_type == 'mc_grad':
        sde_solver_step = integration_step_mc_est_grad
    elif score_type == 'mc_reparam':
        sde_solver_step = integration_step_mc_est_reparam
    elif score_type == 'mc_grad_reparam':
        sde_solver_step = integration_step_mc_est_grad_reparam
    else:
        raise NotImplementedError('Score type {} not supported.'.format(score_type))
    # Deduce the score from the MC estimator
    nabla_log_p_t = lambda y, t, sigma, alpha : get_nabla_log_pt_from_mc_est(score_est, score_type, y, t, sigma, alpha)
    # Solve the SDE with Euler
    if verbose:
        r = trange(0, K-1)
    else:
        r = range(0, K-1)
    for k in r:
        # Run a step of SDE solver
        y = sde_solver_step(y, score_est, ts[k+1], ts[k], sigma, alpha, use_exponential_integrator)
        # Execute callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback(y, ts[k+1])
    # Denoise y_T
    ret = y
    ret += torch.square(sigma) * ts[-1] * nabla_log_p_t(y, ts[-1], sigma, alpha)
    return ret / alpha.alpha(ts[-1])

def sample_y_init(shape, sigma, epsilon, alpha, device, langevin_init=False, n_langevin_steps=10, score_est=None, score_type=None, target_mean=None):
    """Sample the generalized stochastic localization algorithm

    Args:
        shape (tuple of int): Shape of the data
        sigma (float): Value of sigma
        epsilon (float): Value of epsilon
        alpha (Alpha): Object containing the alpha(t) implementation
        device (torch.device): Device used for computations
        langevin_init (bool): Whether to run a short Lanegevin chain from the initialization (default is False)
        n_langevin_steps (int): Number of Langevin steps to run (default is 10)
        score_est (function): Score estimator
        score_type (str): Type of score estimator (in 'nabla_log_p_t', 'mc', 'mc_grad', 'mc_reparam', 'mc_grad_reparam') (default is None)
        target_mean (torch.Tensor of shape shape[1:]): Estimation of the target mean

    Returns:
        y_init (torch.Tensor of shape shape): Initialization for the generalized stochastic localization algorithm
    """

    # Get the mean of the target distribution
    if target_mean is not None:
        target_mean_ = target_mean.to(device)
    else:
        target_mean_ = torch.zeros(shape[1:], device=device)
    # Set y
    y = alpha.alpha(epsilon) * target_mean_ + sigma * math.sqrt(epsilon) * torch.randn(shape, device=device)
    # Refine with Langevin or not
    if langevin_init:
        # Get the score at time epsilon
        score = lambda x : get_nabla_log_pt_from_mc_est(score_est, score_type, x, epsilon, sigma, alpha)
        # Set the step size (the step size is 1/L where L is the lipschitz constant of the score)
        step_size = torch.square(sigma) * epsilon / 2.
        # Run ULA chain
        y = ula_mcmc(y.clone(), step_size, score, n_langevin_steps)
        return y.detach()
    else:
        return y