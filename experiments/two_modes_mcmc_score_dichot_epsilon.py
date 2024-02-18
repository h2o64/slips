# Libraries
import pickle
import torch
import math
from tqdm import trange
from slips.samplers.sto_loc import sample_y_init, sto_loc_algorithm
from slips.samplers.alphas import *
from two_modes_utils import make_target, compute_relative_weights, compute_metrics_mog
from slips.samplers.mcmc import MCMCScoreEstimator

# Binary search


def binary_search(metric, low, high, target_value, n_attemps):
    """Basic binary search

    Args:
            metric (function): Increasing function to evaluate
            low (float): Lower end
            high (float): Higher end
            target_value (float): Target value
            n_attemps (int): Number of attemps

    Returns:
            idx (float): Target idx
    """

    for _ in trange(n_attemps):
        # Get the middle point
        mid = (low + high) / 2.
        ret = metric(mid)
        # Check the different conditions
        if ret < target_value:
            low = mid
        else:
            high = mid
    return (low + high) / 2.


def metric(epsilon, device, batch_size, target, dim, score_est, score_type,
           sigma, alpha, K, T, epsilon_end, return_metrics=False):
    """Run the experiment

    Args:
            epsilon (float): Initial integration time
            device (torch.Device): Device to use for the computations
            batch_size (int): Number of samples
            target (slips.distributions.*): Target distribution
            score_est (function): Score estimator
            score_type (str): Type of score estimator
            sigma (float): Value of sigma
            alpha (Alpha): Alpha value
            K (int): Computationnal budget
            T (float): Theoretical end time
            epsilon_end (float): Gap between the theoretical end time and the practical one
            return_metrics (bool): Whether to return the metrics (default is False)

    Returns:
            if return_metrics:
                    metrics (dict): All the GMM metrics
            else:
                    weight (float): Relative weight estimate

    """

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

    # Compute the relative weights
    weight = compute_relative_weights(target.means[0], target.means[1], samples) - (2. / 3.)
    if return_metrics:
        target_samples = target.sample((batch_size,))
        return compute_metrics_mog(target.means, target.covs, samples, target_samples=target_samples)
    else:
        return weight


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
    parser.add_argument('--n_mcmc_chains', type=int, default=4)
    parser.add_argument('--n_mcmc_samples', type=int, default=32)
    parser.add_argument('--use_reparametrized', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_gradient_observable', action=argparse.BooleanOptionalAction)
    parser.add_argument('--epsilon_min', type=float, default=1e-4)
    parser.add_argument('--T_classic', type=float, default=150.)
    parser.add_argument('--K', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--n_attemps', type=int, default=20)
    args = parser.parse_args()

    # Make a Pytorch device
    device = torch.device('cuda')

    # Set the seed
    torch.manual_seed(args.seed)

    # Make the target
    target = make_target(device, args.dim, args.a)
    sigma = torch.tensor(math.sqrt(((4 / 3) * args.a)**2 + 0.05))

    # Get the score of the target distribution
    def target_log_prob_and_grad(y):
        y_ = torch.autograd.Variable(y, requires_grad=True)
        log_prob_y = target.log_prob(y_)
        return log_prob_y, torch.autograd.grad(log_prob_y.sum(), y_)[0].detach()

    # Classic kwargs for MCMCScoreEstimator
    kwargs = {
        'step_size': args.step_size,
        'n_mcmc_samples': args.n_mcmc_samples,
        'log_prob_and_grad': target_log_prob_and_grad,
        'use_last_mcmc_iterate': True,
        'n_mcmc_chains': args.n_mcmc_chains,
        'keep_mcmc_length': int(0.5 * args.n_mcmc_samples)
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
        T = args.T_classic
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
        epsilon_end = alpha.T - alpha.g_inv(alpha_classic.g(args.T_classic))
    else:
        epsilon_end = 0.0

    # Build the metric
    def metric_(epsilon): return metric(epsilon, device, args.batch_size, target,
                                        args.dim, score_est, score_type, sigma, alpha, args.K, T, epsilon_end)

    # Run the search
    epsilon = binary_search(metric_, args.epsilon_min, T - epsilon_end, 0.0, args.n_attemps)
    metrics = metric(
        epsilon,
        device,
        args.batch_size,
        target,
        args.dim,
        score_est,
        score_type,
        sigma,
        alpha,
        args.K,
        T,
        epsilon_end,
        return_metrics=True)

    # Dump the result
    filename = 'mcmc_score_dichot_search_alpha_{}_reparam_{}_grad_{}_{}.pkl'.format(
        args.alpha_type, str(bool(args.use_reparametrized)), str(bool(args.use_gradient_observable)), args.seed)
    with open('{}/{}'.format(args.results_path, filename), 'wb') as f:
        pickle.dump({'epsilon': epsilon, **metrics}, f)
