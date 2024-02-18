# Compute the metrics for each target given samples

# Libraries
import argparse
import torch
from tqdm import tqdm
import glob
import pickle
from benchmark_utils import compute_metrics

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--results_path', type=str)
args = parser.parse_args()

# Get all the benchmarks
filepaths = list(glob.glob(args.input_path + '/*.pkl'))

# Set the device
device = torch.device('cuda')

# Extract informations
def data_extractor(filename):
	for name in ['ais','smc','mnm','rdmc','sto_loc_classic','sto_loc_geometric_1_1','sto_loc_geometric_2_1']:
		if name in filename:
			algorithm_name = name
	for name in ['8gaussians', 'rings', 'funnel', 'two_modes_dim_8', 'two_modes_dim_16', 'two_modes_dim_32', 'two_modes_dim_64', 'two_modes_dim_128', 'ionosphere', 'sonar',
		'phi_four_b_0.00e+00_dim_100', 'phi_four_b_2.50e-02_dim_100', 'phi_four_b_5.00e-02_dim_100', 'phi_four_b_7.50e-02_dim_100', 'phi_four_b_1.00e-01_dim_100'
	]:
		if name in filename:
			target_name = name
	return algorithm_name, target_name

for filepath in tqdm(filepaths):
	# Get the filename
	filename = filepath.split('/')[-1]
	# Get the algorithm name and target type
	algorithm_name, target_name = data_extractor(filename)
	# Load the data
	with open(filepath, 'rb') as f:
		l = pickle.load(f)
	# Compute the metrics
	if ('params' in l[0]) or (algorithm_name in ['smc','ais']):
		for i in range(len(l)):
			if algorithm_name == 'ais':
				samples, weights = l[i]['results']
				l[i]['metrics'] = compute_metrics(device, target_name, samples.to(device), weights=weights.to(device))
			else:
				l[i]['metrics'] = compute_metrics(device, target_name, l[i]['results'].to(device))
			del l[i]['results']
		# Save the data
		with open(args.results_path + '/' + filename, 'wb') as f:
			pickle.dump(l, f)
