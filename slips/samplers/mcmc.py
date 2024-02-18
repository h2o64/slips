# Markov Chain Monte Carlo algorithms

# Libraries
import math
import torch
from .utils import (
    sample_multivariate_normal_diag,
    log_prob_multivariate_normal_diag,
    heuristics_step_size,
    heuristics_step_size_vectorized
)


def ula_mcmc(x0, step_size, score, n_steps, return_intermediates=False, return_intermediates_gradients=False):
    """Perform multiple steps of the ULA algorithm

        X_{k+1} = X_k + steps_size * score(X_k) + sqrt(2 * step_size) * Z_k

    Args:
        x0 (torch.Tensor of shape (batch_size, *data_shape)): Initial sample
        step_size (float): Step size for Langevin
        score (function): Gradient of the log-likelihood of the target distribution
        n_steps (int): Number of steps of the algorithm
        return_intermediates (bool): Whether to return intermediates states
        return_intermediates_gradients (bool): Whether to return intermediates gradients
            (only is return_intermediates is True)

    Returns:
        x (torch.Tensor of shape (batch_size, *data_shape)): Final sample of the Langevin chain
    """

    x = x0
    if return_intermediates:
        xs = torch.empty((n_steps, *x.shape), device=x.device)
        if return_intermediates_gradients:
            grad_xs = torch.empty_like(xs)
    for i in range(n_steps):
        grad_x = score(x)
        x += step_size * grad_x + torch.sqrt(2. * step_size) * torch.randn_like(x)
        if return_intermediates:
            xs[i] = x.clone()
            if return_intermediates_gradients and i >= 1:
                grad_xs[i - 1] = grad_x.clone()
    if return_intermediates:
        if return_intermediates_gradients:
            grad_xs[-1] = score(x).clone()
            return xs, grad_xs
        else:
            return xs
    else:
        return x


def mala_mcmc(
        x0,
        step_size,
        log_prob_and_grad,
        n_steps,
        per_chain_step_size=True,
        return_intermediates=False,
        return_intermediates_gradients=False,
        target_acceptance=0.75):
    """Perform multiple steps of the MALA algorithm

        X_{k+1} = X_k + steps_size * score(X_k) + sqrt(2 * step_size) * Z_k

    Args:
        x0 (torch.Tensor of shape (batch_size, *data_shape)): Initial sample
        step_size (float): Step size for Langevin
        score (function): Gradient of the log-likelihood of the target distribution
        n_steps (int): Number of steps of the algorithm
        per_chain_step_size (bool): Use a per chain step size (default is True)
        return_intermediates (bool): Whether to return intermediates steps
        target_acceptance (float): Default is 0.75
        return_intermediates_gradients (bool): Whether to return intermediates gradients
            (only is return_intermediates is True)

    Returns:
        x (torch.Tensor of shape (batch_size, *data_shape)): Final sample of the Langevin chain
    """

    sum_indexes = (1,) * (len(x0.shape) - 1)
    x = x0
    log_prob_x, grad_x = log_prob_and_grad(x)
    if return_intermediates:
        xs = torch.empty((n_steps, *x.shape), device=x.device)
        if return_intermediates_gradients:
            grad_xs = torch.empty_like(xs)
    # Reshape the step size if it hasn't been done yet
    if per_chain_step_size and not (isinstance(step_size, torch.Tensor) and len(step_size.shape) > 0):
        step_size = step_size * torch.ones((x.shape[0], *(1,) * (len(x.shape) - 1)), device=x.device)
    for i in range(n_steps):
        # Sample the proposal
        x_prop = sample_multivariate_normal_diag(
            batch_size=x.shape[0],
            mean=x + step_size * grad_x,
            variance=2.0 * step_size
        )
        # Compute log-densities at the proposal
        log_prob_x_prop, grad_x_prop = log_prob_and_grad(x_prop)
        # Compute the MH ratio
        with torch.no_grad():
            joint_prop = log_prob_x_prop - \
                log_prob_multivariate_normal_diag(
                    x_prop,
                    mean=x + step_size * grad_x,
                    variance=2.0 * step_size,
                    sum_indexes=sum_indexes)
            joint_orig = log_prob_x - log_prob_multivariate_normal_diag(x,
                                                                        mean=x_prop + step_size * grad_x_prop,
                                                                        variance=2.0 * step_size,
                                                                        sum_indexes=sum_indexes)
        # Acceptance step
        log_acc = joint_prop - joint_orig
        mask = torch.log(torch.rand_like(log_prob_x_prop, device=x.device)) < log_acc
        x.data[mask] = x_prop[mask]
        log_prob_x.data[mask] = log_prob_x_prop[mask]
        grad_x.data[mask] = grad_x_prop[mask]
        # Update the step size
        if per_chain_step_size:
            step_size = heuristics_step_size_vectorized(step_size,
                                                        torch.minimum(torch.ones_like(log_acc), torch.exp(log_acc)), target_acceptance=target_acceptance)
        else:
            step_size = heuristics_step_size(step_size,
                                             torch.minimum(torch.ones_like(log_acc), torch.exp(log_acc)).mean(), target_acceptance=target_acceptance)
        # Save the sample
        if return_intermediates:
            xs[i] = x.clone()
            if return_intermediates_gradients:
                grad_xs[i] = grad_x.clone()
    if return_intermediates:
        if return_intermediates_gradients:
            return xs, grad_xs, step_size
        else:
            return xs, step_size
    else:
        return x, step_size


def underdamped_langevin_mcmc(x0, step_size, friction, lipschitz_cte, score, n_steps, return_intermediates=False):
    """Perform multiple steps of the Underdamped Langevin algorithm by Sachs et al. (2017)

    Args:
        x0 (torch.Tensor of shape (batch_size, *data_shape)): Initial sample
        step_size (float): Step size for Langevin
        friction (float): Friction coefficient
        lipschitz_cte (float): Lipschitz constant
        score (function): Gradient of the log-likelihood of the target distribution
        n_steps (int): Number of steps of the algorithm
        return_intermediates (bool): Whether to return intermediates steps

    Returns:
        x (torch.Tensor of shape (batch_size, *data_shape)): Final sample of the Langevin chain
    """

    # Usefull constants
    u = 1. / lipschitz_cte
    zeta1 = torch.exp(-friction)
    zeta2 = torch.exp(-2 * friction)
    # Initial values
    v = torch.zeros_like(x0)
    x = x0
    if return_intermediates:
        xs = torch.empty((n_steps, *x.shape), device=x.device)
    # MCMC steps
    for i in range(n_steps):
        x = x + step_size * v / 2  # x_{t+1}
        psi = score(x)
        v = v + u * step_size * psi / 2  # v_{t+1}
        v = zeta1 * v + u * step_size * psi / 2 + math.sqrt(u * (1 - zeta2)) * torch.randn_like(x)  # v_{t+1}
        x = x + step_size * v / 2  # y_{t+1}
        if return_intermediates:
            xs[i] = x.clone()
    if return_intermediates:
        return xs
    else:
        return x


class MCMCScoreEstimator:
    """Markov-Chain-Monte-Carlo scores estimators"""

    def __init__(self, step_size, n_mcmc_samples, reparametrize=False, return_nabla_log_p_t=False, log_prob_and_grad=None, log_prob_grad=None, target_acceptance=0.75,
                 use_gradient_observable=False, use_last_mcmc_iterate=True, n_mcmc_chains=1, keep_mcmc_length=-1):
        """Constructor

        Args:
            step_size (float): Step size for the Langevin MCMC
            n_mcmc_samples (int): Number of MCMC samples to run
            reparametrize (bool): Whether to reparametrize the posterior (default is False)
            return_nabla_log_p_t (bool): Whether to directly return an estimation of nabla_log_p_t or not (default is False)
            log_prob_and_grad (function): Function returning the log-likelihood of the target and its gradient
                (set to None if you want to use ULA)
            log_prob_grad (function): Function returning the log-likelihood of the target
                (set to None if you want to use MALA)
            target_acceptance (float): Target acceptance rate for MALA (default is 0.75)
            use_gradient_observable (bool): Whether to use the gradient of the target as an observable
                (default is False)
            use_last_mcmc_iterate (bool): Initialize MCMC chain with the last MCMC result (default is True)
            n_mcmc_chains (int): Number of parrallel MCMC chains (default is 1)
            keep_mcmc_length (int): Number of MCMC samples to use (default is -1 which means that everything is kept)
        """

        self.reparametrize = reparametrize
        self.default_step_size = step_size
        self.step_size = step_size
        self.use_last_mcmc_iterate = use_last_mcmc_iterate
        self.last_mcmc_iterate = None
        self.return_nabla_log_p_t = return_nabla_log_p_t
        self.n_mcmc_samples = n_mcmc_samples
        self.keep_mcmc_length = self.n_mcmc_samples if keep_mcmc_length < 0 else keep_mcmc_length
        self.n_mcmc_chains = n_mcmc_chains
        self.target_acceptance = target_acceptance
        self.log_prob_and_grad = log_prob_and_grad
        self.log_prob_grad = log_prob_grad
        self.use_gradient_observable = use_gradient_observable

    def reset(self):
        """Reset the step size adaptation and the last mcmc iterate"""
        self.step_size = self.default_step_size
        self.last_mcmc_iterate = None

    def get_log_prob_grad(self, x):
        """Computes the gradient of the target's log-likelihood

        Args:
            x (torch.Tensor): Input states

        Returns:
            grad (torch.Tensor with the same shape as x): Gradients
        """

        shape = x.shape
        if self.log_prob_and_grad is not None:
            ret = self.log_prob_and_grad(x.view((-1, x.shape[-1])))[1]
        else:
            ret = self.log_prob_grad(x.view((-1, x.shape[-1])))
        return ret.view(shape)

    def cond_score_reparam(self, z, y, t, sigma, alpha):
        """Score of p_t(z | y) (with unormalized likelihood if log_prob_and_grad is not None)

        This is
            p_t(z; y) = pi(alpha(t)^{-1} y + (sigma / g(t)) * z) * N(z; 0, I)

        Args:
            x (torch.Tensor of shape (batch_size, *data_shape)): Sampling dimension
            y (torch.Tensor of shape (batch_size, *data_shape)): Observed states
            t (float): Current time
            sigma (float): Value of sigma
            alpha (Alpha): Alpha object

        Returns:
            if self.log_prob_and_grad:
                log_prob (torch.Tensor of shape (batch_size,)): Unormalized log-likelihood of the conditional distribution
            grad (torch.Tensor of shape (batch_size, *data_shape)): Gradient of the log-likelihood of the target distribution
        """

        if self.log_prob_and_grad:
            log_prob_pi, grad_pi = self.log_prob_and_grad((y / alpha.alpha(t)) + (sigma * z / alpha.g(t)))
            log_prob = log_prob_pi - 0.5 * torch.sum(torch.square(z), dim=-1)
            grad = (sigma * grad_pi / alpha.g(t)) - z
            return log_prob, grad
        else:
            grad_pi = self.log_prob_grad((y / alpha.alpha(t)) + (sigma * z / alpha.g(t)))
            return (sigma * grad_pi / alpha.g(t)) - z

    def cond_score_non_reparam(self, x, y, t, sigma, alpha):
        """Score of p_t(x | y) (with unormalized likelihood if log_prob_and_grad is not None)

        This is
            p_t(x; y) = pi(x) * N(y; alpha(t) * x, sigma^2 t I)

        Args:
            x (torch.Tensor of shape (batch_size, *data_shape)): Sampling dimension
            y (torch.Tensor of shape (batch_size, *data_shape)): Observed states
            t (float): Current time
            sigma (float): Value of sigma
            alpha (Alpha): Alpha object

        Returns:
            if self.log_prob_and_grad:
                log_prob (torch.Tensor of shape (batch_size,)): Unormalized log-likelihood of the conditional distribution
            grad (torch.Tensor of shape (batch_size, *data_shape)): Gradient of the log-likelihood of the target distribution
        """

        if self.log_prob_and_grad:
            log_prob_pi, grad_pi = self.log_prob_and_grad(x)
            log_prob = log_prob_pi - 0.5 * \
                torch.sum(torch.square(alpha.alpha(t) * x - y), dim=-1) / (torch.square(sigma) * t)
            grad = grad_pi + (alpha.g(t) / torch.square(sigma)) * ((y / math.sqrt(t)) - alpha.g(t) * x)
            return log_prob, grad
        else:
            grad_pi = self.log_prob_grad(x)
            return grad_pi + (alpha.g(t) / torch.square(sigma)) * ((y / math.sqrt(t)) - alpha.g(t) * x)

    def cond_score(self, w, y, t, sigma, alpha):
        """Score of p_t(x = w| y) or p_t(z = w| y)

        Args:
            w (torch.Tensor of shape (batch_size, *data_shape)): Sampling dimension
            y (torch.Tensor of shape (batch_size, *data_shape)): Observed states
            t (float): Current time
            sigma (float): Value of sigma
            alpha (Alpha): Alpha object

        Returns:
            if self.log_prob_and_grad:
                log_prob (torch.Tensor of shape (batch_size,)): Unormalized log-likelihood of the conditional distribution
            grad (torch.Tensor of shape (batch_size, *data_shape)): Gradient of the log-likelihood of the target distribution
        """

        if self.reparametrize:
            return self.cond_score_reparam(w, y, t, sigma, alpha)
        else:
            return self.cond_score_non_reparam(w, y, t, sigma, alpha)

    def sample(self, x0, step_size, target_fn, manual_n_steps=None):
        """Sample a given target distribution using ULA or MALA

        Args:
            x (torch.Tensor of shape (batch_size, *data_shape)): Input states
            step_size (float): Step size for Langevin
            target_fn (function): Target distribution (only gradient or with log-likelihood)
            manual_n_steps (int): Manual number of MCMC steps (default is None)

        Returns:
            xs (torch.Tensor of shape (n_mcmc_samples, *x.shape)): MCMC samples
        """

        if self.log_prob_and_grad is not None:
            xs, self.step_size = mala_mcmc(
                x0,
                self.step_size,
                target_fn,
                self.n_mcmc_samples if manual_n_steps is None else manual_n_steps,
                return_intermediates=True,
                target_acceptance=self.target_acceptance
            )
            return xs
        else:
            return ula_mcmc(x0, self.step_size, target_fn, self.n_mcmc_samples, return_intermediates=True)

    def __call__(self, y, t, sigma, alpha):
        """Markov-Chain-Monte-Carlo estimator for the score of the noise distribution

            if self.use_gradient_observable is True, the gradient of the target distribution is used as an observable

        Args:
            y (torch.Tensor of shape (batch_size, *data_shape)): Sample for score evaluation
            t (float): Current time
            sigma (float): Value of sigma
            alpha (alphas.Alpha): Alpha object

        Returns:
            sigma_sq_score (torch.Tensor of shape (batch_size, *data_shape)): Score of the noising distribution (times sigma squared)
        """

        # Reshape y if needed
        if self.n_mcmc_chains > 1:
            orig_shape = y.shape
            y = y.unsqueeze(0).repeat((self.n_mcmc_chains, 1, *(1,) * (len(y.shape) - 1))).view((-1, *y.shape[1:]))
        # Initial sample
        if self.use_last_mcmc_iterate and not self.last_mcmc_iterate is None:
            # Use the last MCMC step of the previous call
            x0 = self.last_mcmc_iterate
        else:
            # Initialize with the conditioning point
            if self.reparametrize:
                x0 = torch.zeros_like(y)
            else:
                x0 = y / alpha.alpha(t)
        # Initialize the very first step size of MALA
        if self.log_prob_and_grad is not None and (isinstance(self.step_size, type(
                self.default_step_size))) and (self.step_size == self.default_step_size):
            # Run for a long time
            x0 = self.sample(x0, self.step_size, lambda x: self.cond_score(x, y, t, sigma, alpha),
                             manual_n_steps=50 * self.n_mcmc_samples)[-1].clone()
        # Sample with MCMC starting
        xs = self.sample(x0, self.step_size, lambda x: self.cond_score(x, y, t, sigma, alpha))
        xs = xs[-self.keep_mcmc_length:]
        # Store the last MCMC step
        if self.use_last_mcmc_iterate:
            self.last_mcmc_iterate = xs[-1]
        # Compute the gradient of the samples
        if self.use_gradient_observable:
            if self.reparametrize:
                grad_xs = self.get_log_prob_grad((y / alpha.alpha(t)) + (sigma * xs / alpha.g(t)))
            else:
                grad_xs = self.get_log_prob_grad(xs)
            grad_xs = grad_xs[-self.keep_mcmc_length:]
        # Compute the means
        if self.use_gradient_observable:
            grad_xs_mean = grad_xs.mean(dim=0)
        xs_mean = xs.mean(dim=0)
        if self.n_mcmc_chains > 1:
            if self.use_gradient_observable:
                grad_xs_mean = grad_xs.view((-1, *orig_shape)).mean(dim=0)
            xs_mean = xs.view((-1, *orig_shape)).mean(dim=0)
        # Estimate the score
        if self.return_nabla_log_p_t:
            if self.reparametrize:
                if self.use_gradient_observable:
                    return grad_xs_mean / alpha.alpha(t)
                else:
                    return xs_mean / (sigma * math.sqrt(t))
            else:
                if self.use_gradient_observable:
                    return grad_xs_mean / alpha.alpha(t)
                else:
                    return (alpha.alpha(t) * xs_mean - y) / (torch.square(sigma) * t)
        else:
            if self.use_gradient_observable:
                return grad_xs_mean
            else:
                return xs_mean
