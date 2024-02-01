# Utilitary functions

# Libraries
import torch
import math


def heuristics_step_size(stepsize, mean_acceptance, target_acceptance=0.75, factor=1.03, tol=0.01):
    """Heuristic for adaptative step size"""
    if mean_acceptance - target_acceptance > tol:
        return stepsize * factor
    if target_acceptance - mean_acceptance > tol:
        return stepsize / factor
    return min(stepsize, 1.0)

def heuristics_step_size_vectorized(stepsize, mean_acceptance, target_acceptance=0.75, factor=1.03, tol=0.01):
    """Heuristic for adaptative step size in a vectorized fashion"""
    stepsize = torch.minimum(
        torch.where((mean_acceptance - target_acceptance > tol).view((-1, *(1,) * (len(stepsize.shape)-1))), stepsize * factor, stepsize),
        torch.ones_like(stepsize)
    )
    stepsize = torch.minimum(
        torch.where((target_acceptance - mean_acceptance > tol).view((-1, *(1,) * (len(stepsize.shape)-1))), stepsize / factor, stepsize),
        torch.ones_like(stepsize)
    )
    return stepsize

def sample_multivariate_normal_diag(batch_size, mean, variance):
    """Sample according to multivariate normal with diagonal matrix"""
    z = torch.randn((batch_size, *mean.shape[1:]), device=mean.device)
    if isinstance(variance, torch.Tensor):
        return torch.sqrt(variance) * z + mean
    else:
        return math.sqrt(variance) * z + mean


def log_prob_multivariate_normal_diag(samples, mean, variance, sum_indexes):
    """Evaluate the log density of multivariate normal with diagonal matrix

    WARNING: Single sample along a batch size

    The multiplicative factor
            - 0.5 * dim * torch.log(2.0 * torch.pi * variance)
    was removed.
    """
    ret = -0.5 * torch.sum(torch.square(samples - mean), dim=sum_indexes)
    if isinstance(variance, torch.Tensor) and len(variance.shape) > 0:
        ret /= variance.flatten()
    else:
        ret /= variance
    return ret
