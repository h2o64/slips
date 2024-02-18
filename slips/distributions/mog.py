# Mixture of Gaussians

# Libraries
import torch
import math


def log_prob_mog(y, means, covs, weights):
    """Compute the log-likelihood of a mixture of Gaussians

    Args:
            y (torch.Tensor of shape (batch_size, dim)): Samples to evaluate
            mean (torch.Tensor of shape (n_modes, dim)): Means of each Gaussian
            covs (torch.Tensor of shape (n_modes, dim, dim)): Covariance of each Gaussian
            weights (torch.Tensor of shape (n_modes,)): Weight of each Gaussian

    Returns:
            log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
    """

    # Compute the log_prob
    diff = y.unsqueeze(1) - means.unsqueeze(0)
    log_prob = -torch.matmul(diff.unsqueeze(-2), torch.linalg.solve(covs.unsqueeze(0), diff.unsqueeze(-1)))
    log_prob = log_prob.squeeze(-1).squeeze(-1)
    log_prob -= (y.shape[-1] * math.log(2. * math.pi) + torch.logdet(covs))
    log_prob = 0.5 * log_prob.squeeze(0)
    # Compute the prob
    log_prob += torch.log(weights / weights.sum())
    return torch.logsumexp(log_prob, dim=-1)


class MoG:
    """Distribution of a mixture of Gaussians"""

    def __init__(self, means, covs, weights):
        """Constructor

        Args:
                mean (torch.Tensor of shape (n_modes, dim)): Means of each Gaussian
                covs (torch.Tensor of shape (n_modes, dim, dim)): Covariance of each Gaussian
                weights (torch.Tensor of shape (n_modes,)): Weight of each Gaussian
        """

        self.means = means
        self.covs = covs
        self.weights = weights
        self.covariance_matrices_eye = torch.stack(
            [torch.eye(self.means.shape[-1], device=self.means.device)] * self.weights.shape[0])
        mix = torch.distributions.Categorical(weights)
        comp = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=covs)
        self.dist = torch.distributions.MixtureSameFamily(mix, comp, validate_args=False)

    def sample(self, sample_shape):
        """Sample the distribution

        Args:
                sample_shape (tuple of int): Desired shape for the samples

        Returns:
                samples (torch.Tensor of shape (*sample_shape, dim))
        """

        return self.dist.sample(sample_shape)

    def log_prob(self, values):
        """Evaluate the log-likelihood of the distribution

        Args:
                values (torch.Tensor of shape (*sample_shape, dim)): Samples to evaluate

        Returns:
                log_prob (torch.Tensor of shape sample_shape): Log-likelihood of the samples
        """

        return self.dist.log_prob(values)

    def log_prob_p_t(self, y, t, sigma, alpha):
        """Compute the likelihood of the noised version of the distribution
                Y = alpha(t) * X + sigma * sqrt(t) * Z  with X ~ dist and Z ~ N(0,I)
        Args:
                y (torch.Tensor of shape (batch_size, dim)): Samples to evaluate
                t (float): Current time
                sigma (float): The current noise level
                alpha (Alpha): Object containing alpha details
        Returns:
                log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
        """

        # Compute the covariances
        covs = (alpha.alpha(t)**2) * self.covs.clone() + (torch.square(sigma) * t) * self.covariance_matrices_eye
        # Compute the log_prob
        return log_prob_mog(y, alpha.alpha(t) * self.means, covs, self.weights)

    def mnm_sigma(self):
        """Compute the value of sigma based on Eq (4.3) of "Chain of Log-Concave Markov Chains" (arXiv:2305.19473)

        Returns:
            sigma (float): Value of sigma
        """

        R = float(torch.cdist(self.means.unsqueeze(0), self.means.unsqueeze(0)).max())
        tau = float(torch.sqrt(self.covs.max()))
        return math.sqrt(max(0.0, R**2 - tau**2))


class CircularMixture(MoG):
    """Classic mixture of 8 Gaussians in a circle"""

    def __init__(self, device, radius=10.0, scale=0.7):
        """Constructor

        Args:
                device (torch.device): Device to use for computations
                radius (float): Radius of the circle (default is 10.0)
                scale (float): Scale of the individual Gaussians (default is 0.7)
        """

        means = torch.stack([
            radius * torch.Tensor([
                torch.tensor(i * torch.pi / 4).sin(),
                torch.tensor(i * torch.pi / 4).cos()
            ]) for i in range(8)
        ]).to(device)
        covs = scale * torch.stack([torch.eye(2, device=device)] * 8)
        weights = (1 / 8) * torch.ones((8,), device=device)
        super().__init__(means, covs, weights)
