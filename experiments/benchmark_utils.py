# Utilities for the large benchmark

# Libraries
import math
import torch
import pickle
from slips.distributions.mog import CircularMixture
from slips.distributions.rings import Rings
from slips.distributions.funnel import Funnel
from slips.distributions.logistic_regression import LogisticRegression
from slips.distributions.phifour import PhiFour
from two_modes_utils import make_target

# Algorithms
from slips.samplers.smc import smc_algorithm, init_sample_gaussian, init_log_prob_and_grad_gaussian
from slips.samplers.mc import score_mc_est
from slips.samplers.mcmc import MCMCScoreEstimator
from slips.samplers.rdmc import rdmc_algorithm
from slips.samplers.mnm import make_init, oat_sampler
from slips.samplers.sto_loc import sto_loc_algorithm, sample_y_init
from slips.samplers.alphas import *

# Metrics
from slips.metrics.wasserstein import compute_wasserstein, compute_sliced_wasserstein_fast
from slips.metrics.ks import compute_sliced_ks
from two_modes_utils import compute_relative_weights 

def log_prob_and_grad(target_log_prob, y):
	"""Compute the log-likelihood and score of a distribution

	Args:
		target_log_prob (function): Log-likelihood of the target distribution
		y (torch.Tensor of shape (batch_size, *data_shape)): Evaluation point

	Returns:
		log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood
		grad (torch.Tensor of shape (batch_size, *data_shape)): Gradient of the log_likelihood
	"""

	y_ = torch.autograd.Variable(y, requires_grad=True)
	log_prob_y = target_log_prob(y_)
	return log_prob_y, torch.autograd.grad(log_prob_y.sum(), y_)[0].detach()

# All the targets
target_names = ['8gaussians', 'rings', 'funnel', 'two_modes_dim_8', 'two_modes_dim_16', 'two_modes_dim_32', 'two_modes_dim_64', 'two_modes_dim_128', 'ionosphere', 'sonar',
	'phi_four_b_0.00e+00_dim_100', 'phi_four_b_2.50e-02_dim_100', 'phi_four_b_5.00e-02_dim_100', 'phi_four_b_7.50e-02_dim_100', 'phi_four_b_1.00e-01_dim_100']

# Hyper-parameter ranges
hyper_parameter_ranges_default = {
	'mnm': {
		'step_size' : [0.03, 1.0],
		'friction_eff' : [0.0625, 0.05]
	},
	'rdmc' : {
		'T' : [-math.log(0.95), -math.log(0.9), -math.log(0.8), -math.log(0.7)]
	},
	'sto_loc_classic' : {
		'T' : [150, 300], # Corresponding log-SNR = [5.0, 5.7]
		'epsilon' : [0.1, 0.2, 0.4, 1.0, 1.2], # Corresponding log-SNR = [-2.30, -1.61, -0.92, 0.00, 0.18]
	},
	'sto_loc_geometric_1_1' : {
		'epsilon_end' : [9.90e-03, 6.62e-03], # Corresponding log-SNR = [4.6, 5.0]
		'epsilon' : [0.1, 0.15, 0.20], # Corresponding log-SNR = [-2.2, -1.73, -1.3]
	},
	'sto_loc_geometric_2_1' : {
		'epsilon_end' : [9.80e-03, 6.58e-03], # Corresponding log-SNR = [4.6, 5.0],
		'epsilon' : [0.30, 0.35, 0.45] # Corresponding log-SNR = [-2.05, -1.67, -1.0]
	}
}
hyper_parameter_ranges = { algorithm_name : {} for algorithm_name in hyper_parameter_ranges_default.keys() }
for algorithm_name in ['mnm','rdmc']:
	for target_name in target_names:
		hyper_parameter_ranges[algorithm_name][target_name] = hyper_parameter_ranges_default[algorithm_name]

for algorithm_name in ['sto_loc_classic','sto_loc_geometric_1_1','sto_loc_geometric_2_1']:
	for target_name in ['8gaussians','rings','funnel']:
		hyper_parameter_ranges[algorithm_name][target_name] = hyper_parameter_ranges_default[algorithm_name]

for target_name in list(filter(lambda x : 'phi_four' in x, target_names)):
	hyper_parameter_ranges['sto_loc_classic'][target_name] = {
		'T' : [300, 450], # Corresponding log-SNR = [5.7, 6.1]
		'epsilon' : [0.8, 1.0, 1.2, 1.4, 1.8], # Corresponding log-SNR = [-0.22, 0.00, 0.18, 0.34, 0.59]
	}
	hyper_parameter_ranges['sto_loc_geometric_1_1'][target_name] = {
		'epsilon_end' : [3.32e-03, 2.22e-03], # Corresponding log-SNR = [5.7, 6.1]
		'epsilon' : [0.30, 0.35, 0.40, 0.45], # Corresponding log-SNR = [-0.85, -0.62, -0.41, -0.2]
	}
	hyper_parameter_ranges['sto_loc_geometric_2_1'][target_name] = {
		'epsilon_end' : [3.31e-03, 2.21e-03], # Corresponding log-SNR = [5.7, 6.1]
		'epsilon' : [0.40, 0.45, 0.50, 0.55], # Corresponding log-SNR = [-1.32, -1.0, -0.69, -0.4]
	}

for target_name in ['sonar', 'ionosphere'] + list(filter(lambda x : 'two_modes' in x, target_names)):
	hyper_parameter_ranges['sto_loc_classic'][target_name] = {
		'T' : [150], # Corresponding log-SNR = [5.0]
		'epsilon' : [0.03, 0.05, 0.1, 0.2, 0.4], # Corresponding log-SNR = [-3.51, -3.00, -2.30, -1.61, -0.92]
	}
	hyper_parameter_ranges['sto_loc_geometric_1_1'][target_name] = {
		'epsilon_end' : [6.62e-03], # Corresponding log-SNR = [5.0]
		'epsilon' : [0.03, 0.05, 0.1, 0.15, 0.25], # Corresponding log-SNR = [-3.48, -2.94, -2.2, -1.73, -1.1]
	}
	hyper_parameter_ranges['sto_loc_geometric_2_1'][target_name] = {
		'epsilon_end' : [6.58e-03], # Corresponding log-SNR = [5.0]
		'epsilon' : [0.15, 0.20, 0.25,0.35, 0.45], # Corresponding log-SNR = [-3.63, -3.0, -2.48, -1.67, -1.0]
	}


def run_algorithm(algorithm_name, device, n_samples, target_log_prob_and_grad, target_log_prob, R, tau, dim, params, K, n_mcmc_steps):
	"""Run a given algorithm given target and hyper-parameters

	Args:
		algorithm_name (str): Name of the algorithm (in 'smc','ais','mnm','rdmc','sto_loc_classic','sto_loc_geometric_1_1','sto_loc_geometric_2_1')
		device (torch.Device): Device for computations
		n_samples (int): Number of samples to draw
		target_log_prob_and_grad (function): Log-likelihood and gradient of the target distribution
		target_log_prob (function): Log-likelihood of the target distribution
		R (float): Value of R
		tau (float): Value of tau
		dim (int): Dimensionality
		params (dict): Hyper-parameters for the algorithm
		K (int): Outer-loop computationnal budget
		n_mcmc_steps (int): Number of MCMC steps (inner-loop)

	Returns:
		if algorithm_name == 'ais':
			samples (torch.Tensor of shape (n_samples, *data_shape)): Approximate samples
			weights (torch.Tensor of shape (n_samples,)): Weights
		else:
			samples (torch.Tensor of shape (n_samples, *data_shape)): Approximate samples
	"""

	# SMC
	if algorithm_name == 'smc':
		# Compute sigma
		n_particles = n_samples
		sigma = torch.sqrt(torch.tensor(R**2 + tau**2))
		# Run the algorithm
		return smc_algorithm(n_particles=n_particles,
		    target_log_prob=target_log_prob,
		    target_log_prob_and_grad=target_log_prob_and_grad,
		    init_sample=lambda n : init_sample_gaussian(n, sigma, dim, device),
		    init_log_prob_and_grad=lambda x : init_log_prob_and_grad_gaussian(x, sigma),                  
		    betas=torch.linspace(0.0, 1.0, K),
		    n_mcmc_steps=n_mcmc_steps,
		    verbose=False
		).detach().cpu()
	# AIS
	elif algorithm_name == 'ais':
		# Compute sigma
		n_particles = n_samples
		sigma = torch.sqrt(torch.tensor(R**2 + tau**2))
		# Run the algorithm
		samples, weights = smc_algorithm(n_particles=n_particles,
		    target_log_prob=target_log_prob,
		    target_log_prob_and_grad=target_log_prob_and_grad,
		    init_sample=lambda n : init_sample_gaussian(n, sigma, dim, device),
		    init_log_prob_and_grad=lambda x : init_log_prob_and_grad_gaussian(x, sigma),                  
		    betas=torch.linspace(0.0, 1.0, K),
		    n_mcmc_steps=n_mcmc_steps,
		    use_ais=True,
		    verbose=False
		)
		samples, weights = samples.detach().cpu(), weights.detach().cpu()
		return samples, weights
	# MNM
	elif algorithm_name == 'mnm':
		# Compute the right score
		def sigma_sq_score(y, t, sigma, alpha, n_mc_samples=int(256 * (n_mcmc_steps / 32))):
			return score_mc_est(y, t, sigma, alpha, target_log_prob, return_nabla_log_p_t=True, n_mc_samples=n_mc_samples)
		# Compute sigma
		sigma = torch.sqrt(torch.tensor(R**2 - tau**2))
		# Define M and l
		l = 8
		M = int((K / 8) * (n_mcmc_steps / 32))
		# Compute the initial point
		y1 = make_init((n_samples, dim), sigma=sigma, device=device)
		# Run the algorithm (M = K / 4 to compensate for intermediate IS costs)
		return oat_sampler(y1=y1, sigma=sigma, M=M, n_mcmc_steps=l, sigma_sq_score=sigma_sq_score,
		    step_size=torch.tensor(params['step_size']), friction=torch.tensor(params['friction_eff'] / params['step_size']),
		    make_init=None,
		    verbose=False).detach().cpu()
	# RDMC
	elif algorithm_name == 'rdmc':
		# Compute the initial point
		x_init = torch.randn((n_samples, dim), device=device) * math.sqrt(1. - math.exp(-2. * params['T']))
		# Run the algorithm
		return rdmc_algorithm(x_init=x_init,
			target_log_prob=target_log_prob,
			target_log_prob_and_grad=target_log_prob_and_grad,
			T=params['T'],
			K=K,
			n_warm_start_steps=16,
			n_chains=4,
			n_mcmc_steps=n_mcmc_steps,
			n_mc_samples=128,
			verbose=False
		).detach().cpu()
	# Sto loc
	elif 'sto_loc' in algorithm_name:
		# Compute sigma
		sigma = torch.sqrt(torch.tensor((R / math.sqrt(dim))**2 + tau**2))
		# Prepare the score estimator
		score_est = MCMCScoreEstimator(
			reparametrize=False,
			use_gradient_observable=False,
			step_size=1e-5,
			n_mcmc_samples=n_mcmc_steps,
			log_prob_and_grad=target_log_prob_and_grad,
            n_mcmc_chains=4,
            keep_mcmc_length=int(0.5 * n_mcmc_steps)
		)
		# Define alpha
		if 'classic' in algorithm_name:
			alpha = AlphaClassic()
			epsilon, epsilon_end, T = params['epsilon'], 0.0, params['T']
		elif 'geometric_1_1' in algorithm_name:
			alpha = AlphaGeometric(a=1.0, b=1.0)
			epsilon, epsilon_end, T = params['epsilon'], params['epsilon_end'], 1.0
		elif 'geometric_2_1' in algorithm_name:
			alpha = AlphaGeometric(a=2.0, b=1.0)
			epsilon, epsilon_end, T = params['epsilon'], params['epsilon_end'], 1.0
		else:
			raise ValueError('Alpha type {} not supported.'.format('_'.join(algorithm_name.split('_')[2:])))
		# Sample the initial point
		y_init = sample_y_init((n_samples, dim), sigma=sigma, epsilon=epsilon, alpha=alpha, device=device,
		                      langevin_init=True, score_est=score_est, score_type='mc')
		# Run the stochastic localization algorithm 
		return sto_loc_algorithm(alpha, y_init, K, T, sigma, score_est, score_type='mc',
		    epsilon=epsilon, epsilon_end=epsilon_end, use_exponential_integrator=True,
		    use_logarithmic_discretization=False, use_snr_discretization=True,
		    verbose=False).detach().cpu()
	else:
		raise ValueError('Algorithm {} not implemented.'.format(algorithm_name))


def make_target_dist(dist_name, device):
	"""Make the target distribution

	Args:
		dist_name (str): Name of the target distribution (in '8gaussians','rings','funnel','two_modes_dim_*','ionosphere','sonar','phi_four_b_*_dim_*')
		device (torch.Device): Device to use for the computations

	Returns:
		target_log_prob_and_grad (function): Log-likelihood and gradient of the target distribution
		target_log_prob (function): Log-likelihood of the target distribution
		R (float): Value of R
		tau (float): Value of tau
		dim (int): Dimensionality
	"""

	if dist_name == '8gaussians':
		target = CircularMixture(device)
		target_log_prob_and_grad = lambda x : log_prob_and_grad(target.log_prob, x)
		target_log_prob = target.log_prob
		R = float(torch.linalg.norm(target.means[0], dim=-1))
		tau = float(torch.sqrt(target.covs[0,0,0]))
		dim = 2
	elif dist_name == 'rings':
		target = Rings(device=device)
		target_log_prob_and_grad = lambda x : log_prob_and_grad(target.log_prob, x)
		target_log_prob = target.log_prob
		R = 4.
		tau = 0.15
		dim = 2
	elif dist_name == 'funnel':
		target = Funnel(dim=10, device=device)
		target_log_prob_and_grad = lambda x : log_prob_and_grad(target.log_prob, x)
		target_log_prob = target.log_prob
		R = 2.12
		tau = 0.0
		dim = 10
	elif 'two_modes_dim_' in dist_name:
		dim = int(dist_name.split('_')[-1])
		target = make_target(a=1.0, dim=dim, device=device)
		target_log_prob_and_grad = lambda x : log_prob_and_grad(target.log_prob, x)
		target_log_prob = target.log_prob
		R = (4. / 3.) * math.sqrt(dim)
		tau = float(torch.sqrt(target.covs[0,0,0]))
	elif (dist_name == 'ionosphere') or (dist_name == 'sonar'):
		with open('slips/distributions/datasets/{}.pkl'.format(dist_name), 'rb') as f:
			d = pickle.load(f)
			X, y = d['X_train'], d['y_train']
		target = LogisticRegression(X=X, y=y, device=device)
		target_log_prob_and_grad = lambda x : log_prob_and_grad(target.log_prob, x)
		target_log_prob = target.log_prob
		R = 2.5
		tau = 0.0
		dim = X.shape[-1] + 1
	elif 'phi_four' in dist_name:
		b = float(dist_name.split('_')[3])
		dim = int(dist_name.split('_')[-1])
		target = PhiFour(a=0.1, b=b, dim_grid=dim, dim_phys=1, beta=20.)
		target_log_prob = lambda x : -target.beta * target.U(x)
		target_log_prob_and_grad = lambda x : log_prob_and_grad(target_log_prob, x)
		R = 4.5
		tau = 2e-1
	else:
		raise ValueError('Target distribution {} not found.'.format(dist_name))
	return target_log_prob_and_grad, target_log_prob, R, tau, dim

def compute_metrics(device, dist_name, samples, weights=None):
	"""Compute the metrics given approximate samples under a target distribution

	Args:
		device (torch.Device): Device to use for computations
		dist_name (str): Name of the target distribution
		samples (torch.Tensor of shape (n_samples, *data_shape)): Approximate samples
		weights (torch.Tensor of shape (n_samples,)): Weights (default is None)

	Returns:
		metrics (dict): Dictionnary filled with metrics
	"""

	if dist_name == '8gaussians':
		target = CircularMixture(device)
		ret = { 'w2' : float(compute_wasserstein(target.sample((samples.shape[0],)), samples, weights=weights)) }
	elif dist_name == 'rings':
		target = Rings(device=device)
		ret = { 'w2' : float(compute_wasserstein(target.sample((samples.shape[0],)), samples, weights=weights)) }
	elif dist_name == 'funnel':
		target = Funnel(dim=10, device=device)
		ret = { 'ks_sliced' : float(compute_sliced_ks(target.sample((samples.shape[0],)), samples, weights=weights)) }
	elif 'two_modes_dim_' in dist_name:
		dim = int(dist_name.split('_')[-1])
		target = make_target(a=1.0, dim=dim, device=device)
		w2_sliced = compute_sliced_wasserstein_fast(target.sample((samples.shape[0],)), samples, weights=weights)
		mode_weight = compute_relative_weights(target.means[0], target.means[1], samples, weights=weights)
		mode_weight_error = abs(mode_weight - (2./3.))
		ret = { 'w2_sliced' : w2_sliced, 'mode_weight_error' : mode_weight_error }
	elif (dist_name == 'ionosphere') or (dist_name == 'sonar'):
		with open('slips/distributions/datasets/{}.pkl'.format(dist_name), 'rb') as f:
			d = pickle.load(f)
			X, y = d['X_test'], d['y_test']
		target = LogisticRegression(X=X, y=y, device=device)
		ret = { 'mean_pred_log_prob' : float(target.log_prob(samples).mean()) }
	elif 'phi_four' in dist_name:
		# Get the parameters
		b = float(dist_name.split('_')[3])
		dim = int(dist_name.split('_')[-1])
		target = PhiFour(a=0.1, b=b, dim_grid=dim, dim_phys=1, beta=20.)
		# Compute the weight ratio
		mask = (samples[:, int(dim / 2)] > 0).cpu()
		if weights is None:
			weight_ratio = float(mask.float().mean() / (1. - mask.float().mean()))
		else:
			if mask.sum() == 0:
				weight_ratio = 1e-10
			else:
				weight_ratio = float(weights[mask].sum() / weights[~mask].sum())
		en_diff = -math.log(max(weight_ratio, 1e-20))
		# Compute the target weight ratio
		if b == 0.00e+00:
			target_en_diff = torch.tensor(0.00000000e+00)
		elif b == 2.50e-02:
			target_en_diff = torch.tensor(-7.50320435e+00)
		elif b == 5.00e-02:
			target_en_diff = torch.tensor(-1.49925537e+01)
		elif b == 7.50e-02:
			target_en_diff = torch.tensor(-2.24531708e+01)
		elif b == 1.00e-01:
			target_en_diff = torch.tensor(-2.98685608e+01)
		target_weight_ratio = float(torch.exp(-target_en_diff))
		return {
			'abs_err_weight_ratio' : float(abs(target_weight_ratio - weight_ratio)),
			'relative_err_diff_est' : float((en_diff - target_en_diff) / target_en_diff),
			'en_diff' : en_diff,
			'weight_ratio' : weight_ratio
		}
	else:
		raise ValueError('Target distribution {} not found.'.format(dist_name))
	return ret