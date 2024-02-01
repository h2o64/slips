# Libraries
import pickle
import torch
import math
from tqdm import tqdm
from slips.samplers.sto_loc import sample_y_init, sto_loc_algorithm
from slips.samplers.alphas import *
from two_modes_utils import make_target, compute_metrics_mog
from slips.samplers.mcmc import MCMCScoreEstimator

# Run the experiment
def run_exp(epsilon_range, device, batch_size, target, dim, score_est, score_type, sigma, alpha, K, T, epsilon_end):

	# Browse all the epsilons
	results = []
	target_samples = target.sample((batch_size,))
	for epsilon in tqdm(epsilon_range):

		# Reset the score
		score_est.reset()

		# Sample the initial point
		y_init = sample_y_init((batch_size, dim), sigma=sigma, epsilon=epsilon, alpha=alpha, device=device,
		                      langevin_init=True, score_est=score_est, score_type=score_type)

		# Run the stochastic localization algorithm 
		samples = sto_loc_algorithm(alpha, y_init, K, T, sigma, score_est, score_type=score_type,
		    epsilon=epsilon, epsilon_end=epsilon_end, use_exponential_integrator=True,
		    use_logarithmic_discretization=False, use_snr_discretization=True,
		    verbose=False).detach()

		# Append to the results
		results.append({
			'epsilon' : epsilon,
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
	parser.add_argument('--alpha_type', type=str)
	parser.add_argument('--dim', type=int, default=10)
	parser.add_argument('--a', type=float, default=1.0)
	parser.add_argument('--step_size', type=float, default=1e-5)
	parser.add_argument('--n_mcmc_samples', type=int, default=32)
	parser.add_argument('--n_mcmc_chains', type=int, default=4)
	parser.add_argument('--use_reparametrized', action=argparse.BooleanOptionalAction)
	parser.add_argument('--use_gradient_observable', action=argparse.BooleanOptionalAction)
	parser.add_argument('--T_classic', type=float, default=150.)
	parser.add_argument('--K', type=int, default=1024)
	parser.add_argument('--batch_size', type=int, default=4096)
	parser.add_argument('--epsilon_num', type=int, default=15)
	parser.add_argument('--sigma_factor', type=float, default=1.0)
	args = parser.parse_args()

	# Make a Pytorch device
	device = torch.device('cuda')

	# Set the seed
	torch.manual_seed(args.seed)

	# Make the target
	target = make_target(device, args.dim, args.a)
	sigma = torch.tensor(args.sigma_factor * math.sqrt(((4/3) * args.a)**2 + 0.05))

	# Get the score of the target distribution
	def target_log_prob_and_grad(y):
	    y_ = torch.autograd.Variable(y, requires_grad=True)
	    log_prob_y = target.log_prob(y_)
	    return log_prob_y, torch.autograd.grad(log_prob_y.sum(), y_)[0].detach()

	# Classic kwargs for MCMCScoreEstimator
	kwargs = {
	    'step_size' : args.step_size,
	    'n_mcmc_samples' : args.n_mcmc_samples,
	    'log_prob_and_grad' : target_log_prob_and_grad,
	    'use_last_mcmc_iterate' : True,
	    'n_mcmc_chains' : args.n_mcmc_chains,
	    'keep_mcmc_length' : int(0.5 * args.n_mcmc_samples)
	}

	# Make the
	if bool(args.use_reparametrized):
		if bool(args.use_gradient_observable):
			score_est = MCMCScoreEstimator(reparametrize=True, use_gradient_observable=True, **kwargs)
			score_type = 'mc_grad_reparam'
		else:
			score_est = MCMCScoreEstimator(reparametrize=True, use_gradient_observable=False, **kwargs)
			score_type = 'mc_reparam'
	else:
		if bool(args.use_gradient_observable):
			score_est = MCMCScoreEstimator(reparametrize=False, use_gradient_observable=True, **kwargs)
			score_type = 'mc_grad'
		else:
			score_est = MCMCScoreEstimator(reparametrize=False, use_gradient_observable=False, **kwargs)
			score_type = 'mc'

	# Set alpha to be classic sto loc
	alpha_classic = AlphaClassic()
	if args.alpha_type == 'classic':
		alpha = alpha_classic
		T = args.T_classic * (args.sigma_factor**2)
	elif args.alpha_type == 'geometric_1_1':
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

	# Set epsilon_end
	if args.alpha_type != 'classic':
		epsilon_end = alpha.T - alpha.g_inv(alpha_classic.g(args.T_classic * (args.sigma_factor**2)))
	else:
		epsilon_end = 0.0

	# Get the range of epsilons based on previous research
	if args.alpha_type == 'classic':
		epsilon_range_ref = torch.linspace(0.2, 0.4, args.epsilon_num)
	elif args.alpha_type == 'geometric_1_1':
		epsilon_range_ref = torch.linspace(0.10, 0.35, args.epsilon_num)
	elif args.alpha_type == 'geometric_2_1':
		epsilon_range_ref = torch.linspace(0.25, 0.50, args.epsilon_num)

	# Shit the range
	if args.alpha_type == 'classic':
		epsilon_range = epsilon_range_ref * (args.sigma_factor**2)
	elif args.alpha_type == 'geometric_1_1':
		if args.sigma_factor == 1.0:
			epsilon_range = epsilon_range_ref
		elif args.sigma_factor == 0.5:
			epsilon_range = torch.linspace(0.01, 0.26, args.epsilon_num)
		elif args.sigma_factor == 1.5:
			epsilon_range = epsilon_range_ref + 0.2
	elif args.alpha_type == 'geometric_2_1':
		if args.sigma_factor == 1.0:
			epsilon_range = epsilon_range_ref
		elif args.sigma_factor == 0.5:
			epsilon_range = epsilon_range_ref - 0.2
		elif args.sigma_factor == 1.5:
			epsilon_range = epsilon_range_ref + 0.2

	# Run the experiment
	results = run_exp(epsilon_range, device, args.batch_size, target, args.dim, score_est, score_type, sigma, alpha, args.K, T, epsilon_end)

	# Dump the result
	filename = 'mcmc_score_grid_search_sigma_factor_{:.2e}_alpha_{}_reparam_{}_grad_{}_{}.pkl'.format(args.sigma_factor, args.alpha_type, str(bool(args.use_reparametrized)),
		str(bool(args.use_gradient_observable)), args.seed)
	with open('{}/{}'.format(args.results_path, filename), 'wb') as f:
		pickle.dump(results, f)
