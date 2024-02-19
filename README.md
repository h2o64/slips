<div align="center">

# Stochastic Localization via Iterative Posterior Sampling

[![Preprint](http://img.shields.io/badge/paper-arxiv.2402.10758-B31B1B.svg)](https://arxiv.org/abs/2402.10758)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

## Description

This is the official repository for the paper [Stochastic Localization via Iterative Posterior Sampling](https://arxiv.org/abs/2402.10758).
We consider a general stochastic localization framework and introduce an explicit class of observation processes, associated with flexible denoising schedules. We provide a complete methodology, *Stochastic Localization via Iterative Posterior Sampling* (**SLIPS**), to obtain approximate samples of this dynamics, and as a by-product, samples from the unnormalized target distribution. Our scheme is based on a Markov chain Monte Carlo estimation of the denoiser and comes with detailed practical guidelines. We illustrate the benefits and applicability of SLIPS on several benchmarks, including Gaussian mixtures in increasing dimensions, Bayesian logistic regression and a high-dimensional field system from statistical-mechanics. We experiment multiple tasks : 

- Toy target distributions (8 Gaussians and Rings) (2 dimensions)
- Funnel (10 dimensions)
- Mixture of two Gaussians (from 8 up to 128 dimensions)
- Bayesian Logisitic Regressions (with 34 and 61 dimensions)
- Phi four field system (with 100 dimensions)

This code also contains implementation of the following papers :
- [Reverse Diffusion Monte Carlo](https://openreview.net/forum?id=kIPEyMSdFV) (see [`slips.samplers.rdmc`](./slips/samplers/rdmc.py))
- [Chain of Log-Concave Markov Chains](https://openreview.net/forum?id=yiMB2DOjsR) (see [`slips.samplers.mnm`](./slips/samplers/mnm.py))
- [Annealed Importance Sampling](https://link.springer.com/article/10.1023/A:1008923215028) (see [`slips.samplers.smc`](./slips/samplers/smc.py))
- [Sequential Monte Carlo](https://www.jstor.org/stable/3879283) (see [`slips.samplers.smc`](./slips/samplers/smc.py))

## Installation

The package can be installed as follows
```bash
# Clone the projet
git clone https://github.com/h2o64/slips/
cd slips
# Create a virtual environment
python3 -m venv venv
source venv/bin/active
# Install the dependencies
pip install -r requirements.txt
# Install the package
pip install -e .
```

You can run the experiments using the following command line
```bash
python experiments/launch_benchmark.py --results_path [OUTPUT_FOLDER] -target_type [TARGET_TYPE] --algorithm_name [ALGORITHM_NAME] --seed [SEED]
```

## Notebooks

We provide 4 notebooks to play with [RDMC](<./notebooks/8 Gaussians - RDMC.ipynb>), [OAT](<./notebooks/8 Gaussians - OAT.ipynb>), [SMC/AIS](<./notebooks/8 Gaussians - SMC and AIS.ipynb>) and [SLIPS](<./notebooks/8 Gaussians - SLIPS.ipynb>) on the toy target 8 Gaussians.

## Citations

If this codebase is useful towards other research efforts please consider citing us.

```
@misc{grenioux2024stochastic,
      title={Stochastic Localization via Iterative Posterior Sampling}, 
      author={Louis Grenioux and Maxence Noble and Marylou Gabrié and Alain Oliviero Durmus},
      year={2024},
      eprint={2402.10758},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

## Contribute

We welcome issues and pull requests (especially bug fixes) and contributions.
We will try our best to improve readability and answer questions!