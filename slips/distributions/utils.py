# Helpers for distributions

# Libraries
import torch


class PolarTransform(torch.distributions.transforms.Transform):
    """Polar transformation"""

    domain = torch.distributions.constraints.real_vector
    codomain = torch.distributions.constraints.real_vector
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, PolarTransform)

    def _call(self, x):
        return torch.stack([
            x[..., 0] * torch.cos(x[..., 1]),
            x[..., 0] * torch.sin(x[..., 1])
        ], dim=-1)

    def _inverse(self, y):
        x = torch.stack([
            torch.norm(y, p=2, dim=-1),
            torch.atan2(y[..., 1], y[..., 0])
        ], dim=-1)
        x[..., 1] = x[..., 1] + (x[..., 1] < 0).type_as(y) * (2 * torch.pi)
        return x

    def log_abs_det_jacobian(self, x, y):
        return torch.log(x[..., 0])
