# Libraries
import pickle
import torch
import math
import itertools
from tqdm import tqdm
from slips.samplers.sto_loc import sto_loc_algorithm
from slips.samplers.alphas import *
from two_modes_utils import make_target, compute_metrics_mog

def run_experiment(device, dim, epsilon_range, epsilon_end_range, target, target_samples, sigma, T, K, nabla_log_p_t, batch_size, alpha, em_only):
	"""Run the experiment

	Args:
		device (torch.Device): Device to use for the computations
		dim (int): Dimension of the target
		epsilon_range (torch.Tensor): Ranges of starting time
		epsilon_end_range (torch.Tensor): Ranges of gap between the theoretical end time and the practical one
		target (slips.distributions.*): Target distribution
		target_samples (torch.Tensor of shape (batch_size, *data_shape)): Exact samples from the target distribution
		sigma (float): Value of sigma
		K (int): Computationnal budget
		T (float): Theoretical end time
		nabla_log_p_t (function): Perfect score
		batch_size (int): Number of samples to draw
		alpha (Alpha): Alpha value
		em_only (bool): Whether to force EM algorithm

	Returns:
		results (list of dict): List of metrics
	"""

	# Browse values of T and K
	results = []
	for epsilon, epsilon_end in tqdm(list(itertools.product(epsilon_range, epsilon_end_range))):
		# Get the initial sample
		y_init = sigma * math.sqrt(epsilon) * torch.randn((batch_size, dim), device=device)
		# Run stochastic localization
		samples = sto_loc_algorithm(alpha, y_init, K, T, sigma, nabla_log_p_t, score_type='nabla_log_p_t',
		    epsilon=epsilon, epsilon_end=epsilon_end, use_exponential_integrator=True and not em_only,
		    use_logarithmic_discretization=False, use_snr_discretization=True,
		    verbose=False).detach()
		# Compute the Wasserstein
		results.append({ 'epsilon' : epsilon, 'epsilon_end' : epsilon_end, **compute_metrics_mog(target.means, target.covs, samples, target_samples=target_samples)})
	return results

if __name__ == "__main__":
 
	# Libraries
	import argparse

	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--results_path', type=str)
	parser.add_argument('--seed', type=int)
	parser.add_argument('--alpha_type', type=str)
	parser.add_argument('--dim', type=int, default=10)
	parser.add_argument('--a', type=float, default=3.0)
	parser.add_argument('--batch_size', type=int, default=4096)
	parser.add_argument('--K', type=int, default=2048)
	parser.add_argument('--epsilon_pow_range', type=str, default='7,1')
	parser.add_argument('--epsilon_num', type=int, default=15)
	parser.add_argument('--epsilon_end_pow_range', type=str, default='7,1')
	parser.add_argument('--epsilon_end_num', type=int, default=15)
	parser.add_argument('--em_only', action=argparse.BooleanOptionalAction)
	args = parser.parse_args()

	# Make a Pytorch device
	device = torch.device('cuda')

	# Set the seed
	torch.manual_seed(args.seed)

	# Make the target
	target = make_target(device, args.dim, args.a, centered=False)
	sigma = torch.tensor(math.sqrt(((4/3) * args.a)**2 + 0.05))

	# Make the perfect score
	def nabla_log_p_t(y, t, sigma, alpha):
	    y_ = torch.autograd.Variable(y.clone(), requires_grad=True)
	    log_prob_y = target.log_prob_p_t(y_, t, sigma, alpha)
	    return torch.autograd.grad(log_prob_y.sum(), y_)[0].detach()

	# Set alpha to be classic sto loc
	if args.alpha_type == 'geometric_1_1':
		alpha = AlphaGeometric(a=1.0, b=1.0)
		T = alpha.T
	elif args.alpha_type == 'geometric_1_05':
		alpha = AlphaGeometric(a=1.0, b=0.5)
		T = alpha.T
	elif args.alpha_type == 'geometric_2_1':
		alpha = AlphaGeometric(a=2.0, b=1.0)
		T = alpha.T
	elif args.alpha_type == 'geometric_1_2':
		alpha = AlphaGeometric(a=1.0, b=2.0)
		T = alpha.T
	else:
		raise ValueError('Alpha type {} not found.'.format(args.alpha_type))

	# Sample the target
	target_samples = target.sample((args.batch_size,))

	# Make the epsilon range
	epsilon_range = torch.logspace(*tuple(map(lambda x : -int(x), args.epsilon_pow_range.split(','))), args.epsilon_num)

	# Make the epsilon_end range
	if args.alpha_type == 'geometric_1_1':
		epsilon_end_range = torch.logspace(*tuple(map(lambda x : -int(x), args.epsilon_end_pow_range.split(','))), args.epsilon_end_num)
	else:
		alpha_geo_11 = AlphaGeometric(a=1.0, b=1.0)
		epsilon_end_11_min, epsilon_end_11_max = pow(10, -int(args.epsilon_end_pow_range.split(',')[0])), pow(10, -int(args.epsilon_end_pow_range.split(',')[1]))
		epsilon_end_min = alpha.T - alpha.g_inv(alpha_geo_11.g(alpha_geo_11.T - epsilon_end_11_min))
		epsilon_end_max = alpha.T - alpha.g_inv(alpha_geo_11.g(alpha_geo_11.T - epsilon_end_11_max))
		epsilon_end_range = torch.logspace(math.log10(epsilon_end_min), math.log10(epsilon_end_max), args.epsilon_end_num)

	# Run the experiment
	results = run_experiment(device, args.dim, epsilon_range, epsilon_end_range, target, target_samples, sigma, T, args.K, nabla_log_p_t,
		args.batch_size, alpha, bool(args.em_only))

	# Save the results
	with open('{}/impact_epsilon_finite_time_alpha_type_{}_{}.pkl'.format(args.results_path, args.alpha_type, args.seed), 'wb') as f:
		pickle.dump(results, f)
