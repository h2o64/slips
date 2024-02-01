# Libraries
import pickle
import itertools
import torch
import math
from tqdm import tqdm
from slips.samplers.sto_loc import sto_loc_algorithm
from slips.samplers.alphas import *
from two_modes_utils import make_target, compute_metrics_mog

# Run the experiment
def run_experiment(device, batch_size, dim, target, target_samples, nabla_log_p_t, sigma, K_range, alpha_range, epsilon_classic, T_classic, em_only):

	# Browse all the possibilities
	results = []
	for K, (T, name, alpha) in tqdm(list(itertools.product(K_range, alpha_range))):
		# Set epsilon_end
		if not isinstance(alpha, AlphaClassic):
			alpha_classic = AlphaClassic()
			epsilon_end = T - alpha.g_inv(alpha_classic.g(T_classic))
			epsilon = alpha.g_inv(alpha_classic.g(epsilon_classic))
		else:
			epsilon_end = 0.0
			epsilon = epsilon_classic
		# Sample the initial point
		y_init = sigma * math.sqrt(epsilon) * torch.randn((batch_size, dim), device=device)
		# Run the algorithm
		samples = sto_loc_algorithm(alpha, y_init, K, T, sigma, nabla_log_p_t, score_type='nabla_log_p_t',
		    epsilon=epsilon, epsilon_end=epsilon_end, use_exponential_integrator=True and not em_only,
		    use_snr_discretization=True, verbose=False).detach().to(device)
		# Append the result
		results.append({
		    'alpha' : name, 
		    'K' : K, 
		    **compute_metrics_mog(target.means, target.covs, samples, target_samples=target_samples)
		})
	return results

if __name__ == "__main__":
 
	# Libraries
	import argparse

	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--results_path', type=str)
	parser.add_argument('--seed', type=int)
	parser.add_argument('--dim', type=int, default=10)
	parser.add_argument('--a', type=float, default=1.0)
	parser.add_argument('--batch_size', type=int, default=4096)
	parser.add_argument('--K_pow_range', type=str, default='5,14')
	parser.add_argument('--T_classic', type=float, default=150.)
	parser.add_argument('--epsilon_classic', type=float, default=1e-4)
	parser.add_argument('--em_only', action=argparse.BooleanOptionalAction)
	args = parser.parse_args()

	# Make a Pytorch device
	device = torch.device('cuda')

	# Set the seed
	torch.manual_seed(args.seed)

	# Make the target
	target = make_target(device, args.dim, args.a)
	sigma = torch.tensor(math.sqrt(((4/3) * args.a)**2 + 0.05))

	# Make the perfect score
	def nabla_log_p_t(y, t, sigma, alpha):
	    y_ = torch.autograd.Variable(y.clone(), requires_grad=True)
	    log_prob_y = target.log_prob_p_t(y_, t, sigma, alpha)
	    return torch.autograd.grad(log_prob_y.sum(), y_)[0].detach()

	# Make the ranges
	K_range = [pow(2, i) for i in range(*tuple(map(int, args.K_pow_range.split(','))))]
	alpha_range = [
	    (args.T_classic, 'Classic', AlphaClassic()),
	    (1.0, 'Geometric(a=1.0, b=1.0)', AlphaGeometric(a=1.0, b=1.0)),
	    (1.0, 'Geometric(a=1.0, b=0.5)', AlphaGeometric(a=1.0, b=0.5)),
	    (1.0, 'Geometric(a=2.0, b=1.0)', AlphaGeometric(a=2.0, b=1.0)),
	    (1.0, 'Geometric(a=1.0, b=2.0)', AlphaGeometric(a=1.0, b=2.0))
	]

	# Sample the target
	target_samples = target.sample((args.batch_size,))

	# Run the experiment
	results = run_experiment(device, args.batch_size, args.dim, target, target_samples, nabla_log_p_t,
		sigma, K_range, alpha_range, args.epsilon_classic, args.T_classic, bool(args.em_only))

	# Save the results
	with open('{}/impact_finite_time_{}.pkl'.format(args.results_path, args.seed), 'wb') as f:
		pickle.dump(results, f)
