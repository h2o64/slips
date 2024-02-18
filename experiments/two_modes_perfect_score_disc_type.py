# Libraries
import torch
import math
from slips.samplers.sto_loc import sto_loc_algorithm
from slips.samplers.alphas import *
from two_modes_utils import make_target, compute_metrics_mog, stack_metrics


def run_experiment(device, dim, target, sigma, nabla_log_p_t, batch_size, K, T, epsilon, epsilon_end, alpha, em_only):
    """Run the experiment

    Args:
            device (torch.Device): Device to use for the computations
            dim (int): Dimension of the target
            target (slips.distributions.*): Target distribution
            sigma (float): Value of sigma
            nabla_log_p_t (function): Perfect score
            batch_size (int): Number of samples to draw
            sigma (float): Value of sigma
            K (int): Computationnal budget
            T (float): Theoretical end time
            epsilon (float): Starting time
            epsilon_end (float): Gap between the theoretical end time and the practical one
            alpha (Alpha): Alpha value
            em_only (bool): Whether to force EM algorithm

    Returns:
            results_uniform (list of metrics): Progressive metrics with uniform time
            results_adapted (list of metrics): Progressive metrics with adapted time
    """

    # Get the mean and covs of Y_t
    def get_means_covs(t):
        means_t = alpha.alpha(t) * target.means
        covs_t = (alpha.alpha(t)**2) * target.covs.clone() + (torch.square(sigma) * t) * target.covariance_matrices_eye
        return means_t, covs_t

    # Callback to measure the lipschitz constant
    def callback_wasserstein(y_t, t, results):
        x = target.sample((y_t.shape[0],))
        y_t_true = alpha.alpha(t) * x + sigma * math.sqrt(t) * torch.randn_like(x)
        means_t, covs_t = get_means_covs(t)
        results.append(compute_metrics_mog(means_t, covs_t, y_t, target_samples=y_t_true))

    # Run the stochastic localization algorithm with uniform discretization
    y_init = sigma * math.sqrt(epsilon) * torch.randn((batch_size, dim), device=device)
    results_uniform = []
    _ = sto_loc_algorithm(alpha, y_init, K, T, sigma, nabla_log_p_t, score_type='nabla_log_p_t',
                          epsilon=epsilon, epsilon_end=epsilon_end, use_exponential_integrator=True and not em_only,
                          use_logarithmic_discretization=False, use_snr_discretization=False,
                          callbacks=[lambda y, t: callback_wasserstein(y, t, results_uniform)],
                          verbose=True).detach().cpu()

    # Run the stochastic localization algorithm with adapted discretization
    results_adapted = []
    y_init = sigma * math.sqrt(epsilon) * torch.randn((batch_size, dim), device=device)
    _ = sto_loc_algorithm(alpha, y_init, K, T, sigma, nabla_log_p_t, score_type='nabla_log_p_t',
                          epsilon=epsilon, epsilon_end=epsilon_end, use_exponential_integrator=True,
                          use_logarithmic_discretization=False, use_snr_discretization=True,
                          callbacks=[lambda y, t: callback_wasserstein(y, t, results_adapted)],
                          verbose=True).detach().cpu()

    # Return everything
    return results_uniform, results_adapted


if __name__ == "__main__":

    # Libraries
    import pickle
    import argparse

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--alpha_type', type=str)
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--K', type=int, default=1024)
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
    sigma = torch.tensor(math.sqrt(((4 / 3) * args.a)**2 + 0.05))

    # Make the perfect score
    def nabla_log_p_t(y, t, sigma, alpha):
        y_ = torch.autograd.Variable(y.clone(), requires_grad=True)
        log_prob_y = target.log_prob_p_t(y_, t, sigma, alpha)
        return torch.autograd.grad(log_prob_y.sum(), y_)[0].detach()

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
        epsilon = alpha.g_inv(alpha_classic.g(args.epsilon_classic))
    else:
        epsilon_end = 0.0
        epsilon = args.epsilon_classic

    # Run the experiment
    results_uniform, results_adapted = run_experiment(device, args.dim, target, sigma, nabla_log_p_t,
                                                      args.batch_size, args.K, T, epsilon, epsilon_end, alpha, bool(args.em_only))

    # Stack everything
    results_uniform = stack_metrics(results_uniform)
    results_adapted = stack_metrics(results_adapted)

    # Save the results
    filepath = '{}/adaptive_disc_metrics_alpha_type_{}_{}.pkl'.format(args.results_path, args.alpha_type, args.seed)
    with open(filepath, 'wb') as f:
        pickle.dump({'uniform': results_uniform, 'adapted': results_adapted}, f)
