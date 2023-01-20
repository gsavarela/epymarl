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
from typing import Dict, Union, Tuple
from pathlib import Path
import re
import string
from operator import itemgetter
from collections import defaultdict, OrderedDict
from typing import List

from analysis.stats import standard_error
from src.utils.loaders import loader

import numpy as np

from incense import ExperimentLoader


Array = np.ndarray
ALGO_ID_TO_ALGO_LBL = {
        'NTWA2C': 'NTWA2C',
        'IA2C_NS': 'IA2C',
        'MAA2C_NS': 'MAA2C',
        'NTWQL': 'NTWAQL',
        'IQL_NS': 'IQL',
        'VDN_NS': 'VDN',
}


def file_processor(environment: str, algo: str,  query: Dict):
    root_path = Path(f"results/sacred/{algo}")

    if 'sub_dir' in query:
        root_path = root_path / query.pop('sub_dir')
    root_path = root_path / environment

    steps = defaultdict(list)
    results = defaultdict(list)
    max_rollouts = 41  # Required number of tests
    # max_rollouts = 101  # Required number of tests

    taskname = environment.split(":")[-1].split("-v")[0]
    algoname = algo.upper()
    key = (algoname, taskname)
    sample_size = 0
    for experiment_id in query['query_ids']:
        # Matches every lbforaging task.
        experiment_path = root_path / str(experiment_id)

        print(algoname, experiment_path)
        with (experiment_path / 'metrics.json').open("r") as f:
            data = json.load(f)

        if "test_return_mean" in data:
            _steps = data["test_return_mean"]["steps"]
            _values = data["test_return_mean"]["values"]
            print(f"algo: {algoname}\tsource: {taskname}\tn_points:{len(_values)}")

            # Get at most the 41 first evaluations
            steps[key].append(_steps[:max_rollouts])
            results[key].append(_values[:max_rollouts])
            sample_size += 1

    if sample_size > 0:
        steps[key] = np.vstack(steps[key])
        results[key] = np.vstack(
            results[key]
        )
    return steps, results

def main(
    environment: str,
    algonames: List[str],
    sources: Union[str, List[str]],
    suptitle: str,
):
    """Plots aggregating models by task

    1. Queries algos and aggregates runs
    2. Unite runs and generate group statistics
    3. Plot

    Parameters
    ----------
    environment: str
        String that matches a valid env_args.key
        examples:'rware:rware-tiny-4ag-v1' or 'lbforaging:Foraging-15x15-3p-5f-v1'
    algonames:  List[str]
        A N_ALGO sized list of strings each of which matching C Name
        examples: ['ia2c_ns', 'ntwa2c'] or ['iql_ns', 'ntwql']
    sources: Union[str, List[str]]
        Either a N_ALGO sized list of strings or string.
        examples: 'local' or ['remote', 'local']
        choice:
    """
    if isinstance(sources, str):
        sources = [sources] * len(algonames)
    assert len(sources) == len(algonames)
    assert all([source in ('local', 'remote', 'filesystem') for source in sources])

    for algo, source in zip(algonames, sources):
        if source in ('remote', 'local'):
            _, res = loader(environment, algo, source)
        else:
            raise ValueError()
            

        X = np.vstack(res[(algo.upper(), environment)])
        # Computes maximum returns
        Xbar = np.mean(X, axis=0)
                
        ix = np.argmax(Xbar)
        mux = Xbar[ix]
        stdx = np.std(X[:, ix])
        cix = standard_error(stdx, X.shape[1], 0.95)
        task = environment.split(":")[-1]
        print(f"{task}: Maximum Return: {mux:0.3f} +/- {cix:0.2f}")

        # Computes average returns
        mu_x = np.mean(X)
        std_x = np.std(X)
        ci_x = standard_error(std_x, np.prod(X.shape), 0.95)
        print(f"{task}: Average Return: {mu_x:0.3f} +/- {ci_x:0.2f}")

if __name__ == '__main__':
    ENV = 'mpe:SimpleTag-v0'
    # ENV = 'lbforaging:Foraging-15x15-3p-5f-v1'
    # ENV = 'lbforaging:Foraging-15x15-4p-5f-v1'
    # algosnames = ('iql_ns', 'ntwql', 'vdn_ns')
    algosnames = ('ia2c_ns', 'ntwa2c', 'maa2c_ns')
    source = 'remote'
    # sources = [_q.pop('source') if 'source' in _q else 'remote' for _q in NTWQL_QUERIES[ENV].values()]

    main(ENV, algonames, source)
