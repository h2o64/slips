# Horseshoe Logistic Regression

# Libraries
import torch
from torch.distributions.utils import probs_to_logits
from torch.nn.functional import binary_cross_entropy_with_logits

# Logistic Regression


class LogisticRegression:

    def __init__(self, X, y, device, use_intercept=True, intercept_scale=2.5, threshold=1e-8):
        # Dataset
        self.X = X.float().to(device)
        self.y = y.float().to(device).flatten()
        self.dim = self.X.shape[-1]
        self.threshold = 1e-8
        # Priors
        self.weights_prior = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(
                loc=torch.zeros((self.dim,), device=device),
                scale=torch.ones((self.dim,), device=device)
            ), reinterpreted_batch_ndims=1)
        self.use_intercept = use_intercept
        if use_intercept:
            self.intercept_prior = torch.distributions.Normal(
                loc=torch.tensor(0.0).to(device),
                scale=torch.tensor(intercept_scale).to(device)
            )

    def log_prob(self, params):
        # Ensure the shape of the params
        params = params.reshape((-1, params.shape[-1]))
        # Unpack the parameters
        if self.use_intercept:
            weights, intercept = params[..., :-1], params[..., -1]
        else:
            weights = params
        # Compute the prior
        prior_log_prob = self.weights_prior.log_prob(weights)
        if self.use_intercept:
            prior_log_prob += self.intercept_prior.log_prob(intercept)
        # Compute the likelihood
        probs = torch.special.expit(torch.matmul(self.X, weights.T).T + intercept.unsqueeze(-1))
        probs = torch.clip(probs, self.threshold, 1.0 - self.threshold)
        logits = probs_to_logits(probs, is_binary=True)
        log_prob = - \
            binary_cross_entropy_with_logits(logits, self.y.unsqueeze(
                0).expand((logits.shape[0], -1)), reduction="none").sum(dim=-1)
        return log_prob + prior_log_prob
