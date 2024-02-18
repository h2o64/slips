# Neal's Funnel

# Libraries
import torch
import math

# Normal random variable log-likelihood
sqrt_two_pi = math.sqrt(2 * math.pi)


def log_prob_normal(t, loc, scale):
    return -0.5 * torch.square((t - loc) / scale) - torch.log(scale * sqrt_two_pi)


class Funnel(torch.distributions.Distribution):

    arg_constraints = {
        'dim': torch.distributions.constraints.nonnegative_integer,
        'a': torch.distributions.constraints.positive,
        'b': torch.distributions.constraints.positive
    }
    support = torch.distributions.constraints.real_vector
    has_rsample = True

    def __init__(self, dim, device, a=3.0, b=1.0, validate_args=False):
        self.a = a
        self.sqrt_a = math.sqrt(self.a)
        self.torch_sqrt_a = torch.tensor(self.sqrt_a)
        self.b = b
        self.dim = dim
        self.device = device
        super(Funnel, self).__init__(validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        z = self.sqrt_a * torch.randn((*sample_shape, 1))
        x = torch.exp(self.b * z / 2) * torch.randn((*sample_shape, self.dim - 1))
        return torch.concat((z, x), dim=1).to(self.device)

    def log_prob(self, value):
        return log_prob_normal(value[:, 0], 0, self.torch_sqrt_a) \
            + torch.sum(log_prob_normal(value[:, 1:], 0,
                        torch.unsqueeze(torch.exp(self.b * value[:, 0] / 2), dim=-1)), dim=-1)
