# Compute Wasserstein distances

# Libraries
import math
import torch
import ot as pot
from functools import partial

def compute_sliced_wasserstein_mc(samples1, samples2, n_random_projections=4096, centering=True):
    """Compute the sliced Wasserstein with multiple different random slices

    Args:
        samples1 (torch.Tensor of shape (batch_size, dim)): Samples from the first distribution
        samples2 (torch.Tensor of shape (batch_size, dim)): Samples from the second distribution
        n_random_projections (int): Number of random projections to use (default is 4096)
        centering (bool): Whether to center the distributions

    Return:
        dist (float): The approximated sliced Wasserstein distance
    """

    # Compute the projections
    projs = torch.randn((n_random_projections, samples1.shape[-1]), device=samples1.device)
    projs /= torch.linalg.norm(projs, dim=-1)[..., None]
    # Compute the mean of samples2
    if centering:
        samples1 -= samples1.mean(dim=0, keepdim=True)
        samples2 -= samples2.mean(dim=0, keepdim=True)
    # Project and sort the samples
    samples1_proj = torch.matmul(samples1, projs.T).T.sort(dim=-1).values
    samples2_proj = torch.matmul(samples2, projs.T).T.sort(dim=-1).values
    # Compute the sliced Wasserstein
    return torch.sqrt(torch.mean(torch.square(torch.abs(samples1_proj - samples2_proj))))


def compute_sliced_wasserstein_fast(samples1, samples2, centering=True, weights=None):
    """Compute the sliced Wasserstein distance with Gaussian approximation
    See [1] Kimia Nadjahi, Alain Durmus, Pierre Jacob, Roland Badeau, & Umut Simsekli (2021).
    Fast Approximation of the Sliced-Wasserstein Distance Using Concentration of Random Projections.
    In Advances in Neural Information Processing Systems.

    Args:
        samples1 (torch.Tensor of shape (batch_size, dim)): Samples from the first distribution
        samples2 (torch.Tensor of shape (batch_size, dim)): Samples from the second distribution
        n_random_projections (int): Number of random projections to use (default is 4096)
        centering (bool): Whether to center the distributions
        weights (torch.Tensor of shape (batch_size,)): Optional reweighting of samples2

    Return:
        dist (float): The approximated sliced Wasserstein distance
    """

    dim = samples1.shape[-1]
    if centering:
        samples1 -= samples1.mean(dim=0)
        if weights is not None:
            samples2 -= (weights.unsqueeze(-1) * samples2).mean(dim=0)
        else:
            samples2 -= samples2.mean(dim=0)
    # Approximate SW
    m2_Xc = torch.mean(torch.linalg.norm(samples1, dim=1) ** 2) / dim
    if weights is not None:
        m2_Yc = torch.mean(weights * torch.linalg.norm(samples2, dim=1) ** 2) / dim
    else:
        m2_Yc = torch.mean(torch.linalg.norm(samples2, dim=1) ** 2) / dim
    sw = torch.abs(torch.sqrt(m2_Xc) - torch.sqrt(m2_Yc))
    return sw

def compute_wasserstein(samples0, samples1, method=None, reg=0.05, power=2, weights=None):
    """Compute the Wasserstein (1 or 2) distance (wrt Euclidean cost) between a source and a target
    distributions.

    Stolen from https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/optimal_transport.py#L214

    Args:
        samples0 (torch.Tensor of shape (bs, *dim): Represents the source minibatch
        samples1 (torch.Tensor of shape (bs, *dim): Represents the source minibatch
        method (str): Use exact Wasserstein or an entropic regularization (default is None)
        reg (float): Entropic regularization coefficients (default is 0.05)
        power (int): Power of the Wasserstein distance (1 or 2) (default is 2)
        weights (torch.Tensor of shape (batch_size,)): Optional reweighting of samples1

    Returns
        ret (float): Wasserstein distance
    """

    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a = pot.unif(samples0.shape[0])
    if weights is not None:
        b = weights.cpu().numpy()
    else:
        b = pot.unif(samples1.shape[0])
    if samples0.dim() > 2:
        samples0 = samples0.reshape(samples0.shape[0], -1)
    if samples1.dim() > 2:
        samples1 = samples1.reshape(samples1.shape[0], -1)
    M = torch.cdist(samples0, samples1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret