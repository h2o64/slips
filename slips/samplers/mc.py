# Monte-carlo estimators for the score

# Libraries
import torch
import math


def score_mc_est_grad_based(y, t, sigma, alpha, log_prob_and_grad, return_nabla_log_p_t=False, n_mc_samples=4096):
    """Monte-Carlo estimator for the score of the noise distribution
    This function leverages the gradient of the target distribution.
    Args:
        y (torch.Tensor of shape (batch_size, *data_shape)): Sample for score evaluation
        t (float): Current time
        sigma (float): Value of sigma
        alpha (alphas.Alpha): Alpha object
        log_prob_and_grad (function): Log-likelihood of the target distribution and its gradient
        return_nabla_log_p_t (bool): Whether to directly return an estimation of nabla_log_p_t or not (default is False)
        n_mc_samples (int): Number of Monte-Carlo samples (default is 4096)
    Returns:
        score (torch.Tensor of shape (batch_size, *data_shape)): Return the estimation
    """

    # Sample the standard normal distribution
    data_shape = y.shape[1:]
    z = torch.randn((n_mc_samples, *data_shape), device=y.device)
    z = torch.concat([z, -z], dim=0)
    x = (y.unsqueeze(0) / alpha.alpha(t)) + (sigma / alpha.g(t)) * z.unsqueeze(1)
    log_prob_x, obs = log_prob_and_grad(x.view((-1, *data_shape)))
    log_prob_x = log_prob_x.view((2 * n_mc_samples, -1))
    obs = obs.view((2 * n_mc_samples, -1, *data_shape))
    # Disable gradients from here
    with torch.no_grad():
        # Compute the weights
        weights = torch.nn.functional.softmax(log_prob_x, dim=0)
        # Compute the estimation
        weights = weights.view((*weights.shape, *(1,) * (len(y.shape) - 1)))
    if return_nabla_log_p_t:
        return torch.sum(weights * obs, dim=0) / alpha.alpha(t)
    else:
        return torch.sum(weights * obs, dim=0)


def score_mc_est(y, t, sigma, alpha, log_prob, return_nabla_log_p_t=False, n_mc_samples=4096):
    """Monte-Carlo estimator for the score of the noise distribution

    Args:
        y (torch.Tensor of shape (batch_size, *data_shape)): Sample for score evaluation
        t (float): Current time
        sigma (float): Value of sigma
        alpha (alphas.Alpha): Alpha object
        log_prob (function): Log-likelihood of the target distribution
        return_nabla_log_p_t (bool): Whether to directly return an estimation of nabla_log_p_t or not (default is False)
        n_mc_samples (int): Number of Monte-Carlo samples (default is 4096)
    Returns:
        score (torch.Tensor of shape (batch_size, *data_shape)): Return the estimation
    """

    # Sample the standard normal distribution
    data_shape = y.shape[1:]
    z = torch.randn((n_mc_samples, *data_shape), device=y.device)
    z = torch.concat([z, -z], dim=0)
    x = (y.unsqueeze(0) / alpha.alpha(t)) + (sigma / alpha.g(t)) * z.unsqueeze(1)
    log_prob_x = log_prob(x.view((-1, *data_shape))).view((2 * n_mc_samples, -1))
    # Disable gradients from here
    with torch.no_grad():
        # Compute the weights
        weights = torch.nn.functional.softmax(log_prob_x, dim=0)
        # Compute the observable
        obs = z.unsqueeze(1).expand((-1, y.shape[0], *(-1,) * (len(y.shape) - 1)))
        # Compute the estimation
        weights = weights.view((*weights.shape, *(1,) * (len(y.shape) - 1)))
    if return_nabla_log_p_t:
        return torch.sum(weights * obs, dim=0) / (sigma * math.sqrt(t))
    else:
        return torch.sum(weights * obs, dim=0)
