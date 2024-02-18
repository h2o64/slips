# Launch a full benchmark on an algorithm

# Libraries
import argparse
from tqdm import tqdm
from benchmark_utils import *
from sklearn.model_selection import ParameterGrid

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--target_type', type=str)
parser.add_argument('--algorithm_name', type=str)
parser.add_argument('--n_samples', type=int, default=8192)
parser.add_argument('--K', type=int, default=1024)
parser.add_argument('--n_mcmc_steps', type=int, default=32)
args = parser.parse_args()

# Make a Pytorch device
device = torch.device('cuda')

# Set the seed
torch.manual_seed(args.seed)

# Select a target
target_log_prob_and_grad, target_log_prob, R, tau, dim = make_target_dist(args.target_type, device)

# Make a grid of parameters
if args.algorithm_name in hyper_parameter_ranges:
    params_grid = ParameterGrid(hyper_parameter_ranges[args.algorithm_name][args.target_type])
else:
    params_grid = [None]

# Make the output filename
output_filename = 'target_{}_algo_{}_seed_{}.pkl'.format(args.target_type, args.algorithm_name, args.seed)

# Run the algorithm on the grid
results = []
for params in tqdm(list(params_grid)):
    # Run the algorithm
    ret = run_algorithm(args.algorithm_name, device, args.n_samples, target_log_prob_and_grad,
                        target_log_prob, R, tau, dim, params, args.K, args.n_mcmc_steps)
    # Append to the list of results
    results.append({
        'algorithm_name': args.algorithm_name,
        'target_type': args.target_type,
        'n_samples': args.n_samples,
        'K': args.K,
        'n_mcmc_steps': args.n_mcmc_steps,
        'params': params,
        'results': ret
    })
    # Save the results
    with open('{}/{}'.format(args.results_path, output_filename), 'wb') as f:
        pickle.dump(results, f)
