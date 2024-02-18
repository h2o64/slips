# Libraries
import pickle
import itertools
import torch
import math
from tqdm import tqdm
from slips.samplers.sto_loc import sto_loc_algorithm
from slips.samplers.alphas import *
from two_modes_utils import make_target, compute_metrics_mog


def run_experiment(device, dim, T_range, K_range, target, target_samples,
                   sigma, nabla_log_p_t, batch_size, epsilon, alpha, em_only):
    """Run the experiment

    Args:
            device (torch.Device): Device to use for the computations
            dim (int): Dimension of the target
            T (torch.Tensor): Range of end times
            K_range (torch.Tensor): Range of computationnal budgets
            target (slips.distributions.*): Target distribution
            target_samples (torch.Tensor of shape (batch_size, *data_shape)): Exact samples from the target distribution
            sigma (float): Value of sigma
            nabla_log_p_t (function): Perfect score
            batch_size (int): Number of samples to draw
            epsilon (float): Starting time
            alpha (Alpha): Alpha value
            em_only (bool): Whether to force EM algorithm

    Returns:
            results (list of dict): List of metrics
    """

    # Browse values of T and K
    results = []
    for T, K in tqdm(list(itertools.product(T_range, K_range))):
        # Get the initial sample
        y_init = sigma * math.sqrt(epsilon) * torch.randn((batch_size, dim), device=device)
        # Run stochastic localization
        samples = sto_loc_algorithm(alpha, y_init, K, T, sigma, nabla_log_p_t, score_type='nabla_log_p_t',
                                    epsilon=epsilon, epsilon_end=0.0, use_exponential_integrator=True and not em_only,
                                    use_logarithmic_discretization=False, use_snr_discretization=True,
                                    verbose=False).detach()
        # Compute the Wasserstein
        results.append({'K': K, 'T': T, **compute_metrics_mog(target.means,
                       target.covs, samples, target_samples=target_samples)})
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
    parser.add_argument('--K_range', type=str, default='64,128,256,512,1024')
    parser.add_argument('--T_range', type=str, default='5,10,25,50,100,150')
    parser.add_argument('--epsilon', type=float, default=1e-4)
    parser.add_argument('--em_only', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # Make a Pytorch device
    device = torch.device('cuda')

    # Set the seed
    torch.manual_seed(args.seed)

    # Make the target
    target = make_target(device, args.dim, args.a)
    sigma = torch.tensor(math.sqrt(((4 / 3) * args.a)**2 + 0.05))

    # Make the perfect score
    def nabla_log_p_t(y, t, sigma, alpha):
        y_ = torch.autograd.Variable(y.clone(), requires_grad=True)
        log_prob_y = target.log_prob_p_t(y_, t, sigma, alpha)
        return torch.autograd.grad(log_prob_y.sum(), y_)[0].detach()

    # Set alpha to be classic sto loc
    alpha = AlphaClassic()

    # Make the ranges
    K_range = map(int, args.K_range.split(','))
    T_range = map(float, args.T_range.split(','))

    # Sample the target
    target_samples = target.sample((args.batch_size,))

    # Run the experiment
    results = run_experiment(device, args.dim, T_range, K_range, target, target_samples, sigma, nabla_log_p_t,
                             args.batch_size, args.epsilon, alpha, bool(args.em_only))

    # Save the results
    with open('{}/impact_T_{}.pkl'.format(args.results_path, args.seed), 'wb') as f:
        pickle.dump(results, f)
