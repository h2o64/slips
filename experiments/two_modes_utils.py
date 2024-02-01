# Utilitaries for two modes

# Libraries
import numpy as np
import torch
from slips.metrics.wasserstein import compute_wasserstein, compute_sliced_wasserstein_fast
from slips.distributions.mog import MoG
from sklearn.mixture import GaussianMixture

# Make the target distribution
def make_target(device, dim, a=1.0, centered=True):
    # Make the means
    means = torch.stack([
        -a * torch.ones((dim,), device=device),
        a * torch.ones((dim,), device=device)
    ])
    if centered:
        means += (a/3.) * torch.ones((dim,), device=device)
    # Make the covariances
    covs = torch.stack([0.05 * torch.eye(dim, device=device)] * 2)
    # Make the weights
    weights = torch.FloatTensor([2, 1]).to(device)
    # Return the distribution
    return MoG(means, covs, weights)

# Kullback-Leiber divergence between Gaussians
def compute_kl_gaussians(mean1, cov1, mean2, cov2):
    # Build the two distributions
    dist1 = torch.distributions.MultivariateNormal(loc=mean1, covariance_matrix=cov1,
                                                  validate_args=False)
    dist2 = torch.distributions.MultivariateNormal(loc=mean2, covariance_matrix=cov2,
                                                  validate_args=False)
    # Return the KL
    return float(torch.distributions.kl.kl_divergence(dist1, dist2).cpu())

# Compute the relative weights
def compute_relative_weights(mean1, mean2, samples, weights=None, return_mask=False):
    dist_mode1 = torch.linalg.norm(mean1 - samples, dim=-1)
    dist_mode2 = torch.linalg.norm(mean2 - samples, dim=-1)
    mask = (dist_mode1 < dist_mode2)
    if weights is not None:
        weight = float(((weights * mask.float()).sum() / weights.sum()).cpu())
    else:
        weight = float(mask.float().mean().cpu())
    if return_mask:
        return mask, weight
    else:
        return weight

# Compute MoG metrics with GMM fit
def compute_metrics_mog_gmm(means, covs, samples):
    # Disable gradients
    with torch.no_grad():
        # Make a output dictionnary
        ret = {}
        # Fit the GMM
        try:
            gm = GaussianMixture(n_components=means.shape[0],
                                 weights_init=np.array([2./3., 1./3.]),
                                 means_init=means.cpu().numpy(),
                                 precisions_init=torch.linalg.inv(covs).cpu().numpy())
            gm = gm.fit(samples.cpu().numpy())
        except:
            return {}
        # Get the components
        mask, weight = compute_relative_weights(means[0], means[1], samples, return_mask=True)
        ret['weight_gmm'] = float(np.max(gm.weights_))
        # Compute the means and covs
        means_est = torch.from_numpy(gm.means_).to(samples.device)
        covs_est = torch.from_numpy(gm.covariances_).to(samples.device)
        try:
            ret['kl_mode1_gmm'] = compute_kl_gaussians(means[0], covs[0], means_est[0], covs_est[0])
        except:
            ret['kl_mode1_gmm'] = torch.nan
        try:
            ret['kl_mode2_gmm'] = compute_kl_gaussians(means[1], covs[1], means_est[1], covs_est[1])
        except:
            ret['kl_mode2_gmm'] = torch.nan
        ret['mean_err_mode1_gmm'] = float(torch.linalg.norm(means_est[0] - means[0]).cpu())
        ret['mean_err_mode2_gmm'] = float(torch.linalg.norm(means_est[1] - means[1]).cpu())
        ret['cov_err_mode1_gmm'] = float(torch.linalg.norm(covs[0] - covs_est[0], ord='fro').cpu())
        ret['cov_err_mode2_gmm'] = float(torch.linalg.norm(covs[1] - covs_est[1], ord='fro').cpu())
    return ret

# Compute the various metrics of a Gaussian
def compute_metrics_mog(means, covs, samples, target_samples=None, compute_regularized_w2=False):
    # Disable gradients
    with torch.no_grad():
        # Make a output dictionnary
        ret = {}
        # Compute the mask and weight
        mask, weight = compute_relative_weights(means[0], means[1], samples, return_mask=True)
        ret['weight'] = weight
        # Compute the means and covs
        mean1_est, mean2_est = samples[mask].mean(dim=0), samples[~mask].mean(dim=0)
        cov1_est, cov2_est = samples[mask].T.cov(), samples[~mask].T.cov()
        try:
            ret['kl_mode1'] = compute_kl_gaussians(means[0], covs[0], mean1_est, cov1_est)
        except:
            ret['kl_mode1'] = torch.nan
        try:
            ret['kl_mode2'] = compute_kl_gaussians(means[1], covs[1], mean2_est, cov2_est)
        except:
            ret['kl_mode2'] = torch.nan
        ret['mean_err_mode1'] = float(torch.linalg.norm(mean1_est - means[0]).cpu())
        ret['mean_err_mode2'] = float(torch.linalg.norm(mean2_est - means[1]).cpu())
        ret['cov_err_mode1'] = float(torch.linalg.norm(covs[0] - cov1_est, ord='fro').cpu())
        ret['cov_err_mode2'] = float(torch.linalg.norm(covs[1] - cov2_est, ord='fro').cpu())
        # Compute the sliced Wasserstein
        if target_samples is not None:
            ret['w2_sliced'] = float(compute_sliced_wasserstein_fast(target_samples, samples).cpu())
            if compute_regularized_w2:
                ret['w2'] = compute_wasserstein(target_samples, samples)
    return dict(ret, **compute_metrics_mog_gmm(means, covs, samples))

# Helper function to stack a list of metrics
def stack_metrics(l):
    metrics = list(l[0].keys())
    return {
        metric : torch.FloatTensor(list(map(lambda x : x[metric], l))) for metric in metrics
    }