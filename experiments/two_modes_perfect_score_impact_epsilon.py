# Libraries
import pickle
import torch
import math
from tqdm import tqdm
from slips.samplers.sto_loc import sto_loc_algorithm
from slips.samplers.alphas import *
from two_modes_utils import make_target, compute_metrics_mog

# Run the experiment
def run_experiment(device, dim, epsilon_range, target, target_samples, sigma, T, K, nabla_log_p_t, batch_size, alpha, em_only):

	# Browse values of T and K
	results = []
	for epsilon in tqdm(epsilon_range):
		# Get the initial sample
		y_init = sigma * math.sqrt(epsilon) * torch.randn((batch_size, dim), device=device)
		# Run stochastic localization
		samples = sto_loc_algorithm(alpha, y_init, K, T, sigma, nabla_log_p_t, score_type='nabla_log_p_t',
		    epsilon=epsilon, epsilon_end=0.0, use_exponential_integrator=True and not em_only,
		    use_logarithmic_discretization=False, use_snr_discretization=True,
		    verbose=False).detach()
		# Compute the Wasserstein
		results.append({ 'epsilon' : epsilon, **compute_metrics_mog(target.means, target.covs, samples, target_samples=target_samples)})
	return results

if __name__ == "__main__":
 
	# Libraries
	import argparse

	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--results_path', type=str)
	parser.add_argument('--seed', type=int)
	parser.add_argument('--dim', type=int, default=10)
	parser.add_argument('--a', type=float, default=3.0)
	parser.add_argument('--batch_size', type=int, default=4096)
	parser.add_argument('--K', type=int, default=2048)
	parser.add_argument('--T', type=float, default=12000)
	parser.add_argument('--epsilon_pow_range', type=str, default='15,0')
	parser.add_argument('--epsilon_num', type=int, default=20)
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
	alpha = AlphaClassic()

	# Sample the target
	target_samples = target.sample((args.batch_size,))

	# Make the epsilon range
	epsilon_range = torch.logspace(*tuple(map(lambda x : -int(x), args.epsilon_pow_range.split(','))), args.epsilon_num)

	# Run the experiment
	results = run_experiment(device, args.dim, epsilon_range, target, target_samples, sigma, args.T, args.K, nabla_log_p_t,
		args.batch_size, alpha, bool(args.em_only))

	# Save the results
	with open('{}/impact_epsilon_alpha_type_classic_{}.pkl'.format(args.results_path, args.seed), 'wb') as f:
		pickle.dump(results, f)
