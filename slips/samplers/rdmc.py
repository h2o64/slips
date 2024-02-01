# Reverse Diffusion Monte Carlo from arXiv:2307.02037

# Librairies
import torch
import math
from tqdm import trange
from .mcmc import ula_mcmc, mala_mcmc

def posterior_log_prob_and_grad(t, y, x, T, target_log_prob_and_grad):
    """Compute the posterior distribution of RDMC and its grading

        q_t(y|x) = pi(x) N(exp(T-t)*x, [(1 - exp(-2*(T-t))) / exp(-2*(T-t))] I)

    Args:
        t (torch.Tensor): Current time
        y (torch.Tensor of shape (batch_size, dim)): Evaluation point
        x (torch.Tensor of shape (batch_size, dim)): Conditioning point
        T (torch.Tensor): Limit time
        target_log_prob_and_grad (function): Log-likelihood of the target and its gradient

    Returns:
        log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the posterior
        grad (torch.Tensor of shape (batch_size, dim)): Score of the posterior
    """

    # Evaluate the target
    target_log_prob, target_grad = target_log_prob_and_grad(x)
    # Compute the log_prob
    log_prob = target_log_prob
    log_prob -= 0.5 * torch.sum(torch.square(x - torch.exp(-(T-t))*y), dim=-1) / (1. - torch.exp(-2. * (T-t)))
    # Compute the gradient
    grad = target_grad
    grad += torch.exp(-t) * (torch.exp(-t) * y - x) / (1. - torch.exp(-2. * t))
    # Return everything
    return log_prob, grad

def posterior_importance_sampling(n_chains, t, x, T, target_log_prob, n_mc_samples=128):
    """Sample the importance distribution associated with the posterior of RDMC

    Args:
        n_chains (int): Number of samples to output per batch_size
        t (torch.Tensor): Current time
        x (torch.Tensor for shape (batch_size, dim)): Conditioning point
        T (torch.Tensor): Limit time
        target_log_prob (function): Target log-likelihood
        n_mc_samples (int): Number of particles (default is 128)

    Returns:
        samples (torch.Tensor for shape (n_chains * batch_size, dim)): Approximate samples from the posterior
    """

    # Compute the variance and the mean
    variance = (1. - torch.exp(-2.*(T-t))) / torch.exp(-2.*(T-t))
    mean = torch.exp((T - t)) * x
    # Generate particles
    z = torch.sqrt(variance) * torch.randn((n_mc_samples, *x.shape), device=x.device)
    z += mean.unsqueeze(0)
    # Compute the importane weights
    log_weight = target_log_prob(z.view((-1, *x.shape[1:]))).view((n_mc_samples, -1))
    weights = torch.nn.functional.softmax(log_weight, dim=0)
    # Sample the importance weights
    idx = torch.multinomial(weights.T, n_chains).T
    return torch.gather(z, 0, idx.unsqueeze(-1).expand((-1, -1, z.shape[-1]))).view((-1, z.shape[-1]))

def score_estimation(t, x, T, target_log_prob, target_log_prob_and_grad, step_size, n_mc_samples,
                          n_mcmc_steps, n_chains, warmup_fraction=0.5):
    """Estimate the score of RDMC using IS followed by MCMC

    Args:
        t (torch.Tensor): Current time
        x (torch.Tensor of shape (batch_size, dim)): Evaluation point for the score
        target_log_prob (function): Log-likelihood of the target distribution
        target_log_prob_and_grad (function): Log-likelihood of the target distribution and its gradient
        step_size (torch.Tensor): Step size for MALA
        n_mc_samples (int): Number of particles for IS
        n_mcmc_steps (int): Number of MCMC steps
        n_chains (int): Number of parrallel MCMC chains
        warmup_fraction (float): Warmup proportion (between 0.0 and 1.0 stricly) (default is 0.5)

    Returns:
        score (torch.Tensor of shape (batch_size, dim)): Score at time t and state x
        step_size (torch.Tensor): Updated step size for MALA
    """

    # Sample the importance distribution associated to the posterior
    y_langevin_start = posterior_importance_sampling(n_chains, t, x, T, target_log_prob, n_mc_samples)
    # Run Langevin on the posterior from the IS warm-start
    x_reshaped = x.unsqueeze(0).repeat((n_chains, 1, 1)).view((-1, x.shape[-1]))
    current_posterior_log_prob_and_grad = lambda y : posterior_log_prob_and_grad(t, y, x_reshaped,
        T, target_log_prob_and_grad)
    ys_langevin, step_size = mala_mcmc(y_langevin_start, step_size, current_posterior_log_prob_and_grad,
                            n_mcmc_steps, per_chain_step_size=True, return_intermediates=True)
    ys_langevin = ys_langevin[-int(warmup_fraction * n_mcmc_steps):]
    ys_langevin = ys_langevin.view((int(warmup_fraction * n_mcmc_steps) * n_chains, -1, x.shape[-1]))
    # Compute the approximate score
    return -(x - torch.exp(-(T-t)) * ys_langevin.mean(dim=0)) / (1. - torch.exp(-2.0 * (T - t))), step_size

def rdmc_algorithm(x_init, target_log_prob, target_log_prob_and_grad, T, K, n_warm_start_steps=50, n_chains=10,
                   n_mcmc_steps=10, n_mc_samples=128, verbose=False):
    """Run the RDMC algorithm

    Args:
        x_init (torch.Tensor of shape (batch_size, dim)): Initial point (should be drawn from centered Gaussian
            with covariance (1. - math.exp(-2. * T)) I)
        target_log_prob (function): Log-likehood of the target distribution
        target_log_prob_and_grad (function): Log-likelihood of the target distribution and its gradient
        T (float): Limit time
        K (int): Number of discretization steps for the SDE
        n_warm_start_steps (int): Number of ULA steps for the initial warm start (default is 50)
        n_chains (int): Number of parrallel MCMC chains for score estimation (default is 10)
        n_mcmc_steps (int): Number of MCMC steps for each chain for the score estimation (default is 10)
        n_mc_samples (int): Number of particles in the IS warm-start of the MCMC chains (defalt is 128)
        verbose (bool): Whether to display a progress bar during sampling (default is False)

    Returns:
        samples (torch.Tensor of shape (batch_size, dim)): Approximate samples from the target distribution
    """

    # Sample the reverse noising process
    x = x_init.clone()
    # The initial distribution can be approximated by a centered Gaussian with covariance (1. - math.exp(-2. * T)) I
    step_size = torch.tensor(1. - math.exp(-2. * T)) / 2.
    # Run a Langevin sampler to better approximate the initial distributions
    init_score = lambda x : score_estimation(torch.tensor(0.0), x, T, target_log_prob, target_log_prob_and_grad,
        step_size, n_mc_samples, n_mcmc_steps, n_chains)[0]
    x = ula_mcmc(x.clone(), step_size, init_score, n_warm_start_steps)
    # Loop over the times
    ts = torch.linspace(0.0, T, K)
    if verbose:
        r = trange(K-1)
    else:
        r = range(K-1)
    for i in r:
        # Get the current time
        t = ts[i]
        delta = ts[i+1] - ts[i]
        # Compute the score
        score, step_size = score_estimation(t, x, T, target_log_prob, target_log_prob_and_grad, step_size, n_mc_samples,
                                                 n_mcmc_steps, n_chains)
        # Update x using the estimated score
        x = torch.exp(delta) * x + (torch.exp(delta) - 1.) * score
        x += torch.sqrt(torch.exp(2. * delta) - 1.) * torch.randn_like(x)
    return x