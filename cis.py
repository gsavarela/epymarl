"""Confidence maximum and average returns intervals [1]

 Maximum returns: For each algorithm, we identify the evaluation timestep
 during training in which the algorithm achieves the highest average
 evaluation returns across five random seeds. We report the average returns
 and the 95% confidence interval across five seeds from this evaluation
 timestep.

 Average returns: We also report the average returns achieved throughout all
 evaluations during training. Due to this metric being computed over all
 evaluations executed during training, it considers learning speed besides 
 final achieved returns.


Reference
--------
[1] Papoudakis, Georgios and Christianos, Filippos and Sch\"{a}fer, Lukas and Albrecht, Stefano 
"Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks", 
Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021
"""
import json
from pathlib import Path

from utils import standard_error
import numpy as np

BASE_PATH = Path('results/sacred/maa2c_ns/')
for base_path in BASE_PATH.glob('lbforaging*'):
    test_returns = []
    sample_size = 0
    for path in base_path.rglob('metrics.json'):
        with path.open('r') as f:
            data = json.load(f)

        if 'test_return_mean' in data:
            values = data['test_return_mean']['values']
            print(f'source: {path.parent} n_points:{len(values)}')
            test_returns.append(data['test_return_mean']['values'])
            sample_size += 1
    X = np.vstack(test_returns)


    # Computes maximum returns
    test_return_mean = np.mean(X, axis=0)
    i_x = np.argmax(test_return_mean)
    mu_x = test_return_mean[i_x]
    std_x = np.std(X[:, i_x])
    ci_x = standard_error(std_x, sample_size, 0.95)
    task = base_path.stem.split(':')[-1]
    print(f'{task}: Maximum Return: {mu_x:0.3f} +/- {ci_x:0.2f}')

    # Computes average returns
    mu_x = np.mean(X)
    std_x = np.std(X)
    ci_x = standard_error(std_x, np.prod(X.shape), 0.95)
    task = base_path.stem.split(':')[-1]
    print(f'{task}: Average Return: {mu_x:0.3f} +/- {ci_x:0.2f}')


