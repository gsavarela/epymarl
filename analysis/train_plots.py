from pathlib import Path
from operator import itemgetter
from collections import defaultdict
from typing import List

from analysis.stats import standard_error
from src.utils.loaders import loader
from src.utils.plots import task_plot

import numpy as np
# legends for multiple x-axis

def main(
    environment: str,
    algonames: List[str],
    suptitle: str = ''
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
    suptitle: str, default ''
        The superior title's subtitle
    """
    title = ''
    steps = defaultdict(list)
    results = defaultdict(list)

    # 1. Queries algos and aggregates runs
    for algo in algonames:
        _steps, _results = loader(environment, algo, hypergroup=False)
            
        steps.update(_steps)
        results.update(_results)

        taskname = environment
        title = taskname
        if len(suptitle) > 1:
            title = f"{taskname} ({suptitle})"
        

    # 2. Unite runs and generate group statistics
    algo_task_names = sorted(sorted([*results.keys()], key=itemgetter(1)), key=itemgetter(0))

    # Makes a plot per task
    xs = defaultdict(list)
    mus = defaultdict(list)
    std_errors = defaultdict(list)
    for algo_task_name in algo_task_names:
        # Computes average returns
        xs[algo_task_name] = np.mean(steps[algo_task_name], axis=0)
        mus[algo_task_name] = np.mean(results[algo_task_name], axis=0)
        std = np.std(results[algo_task_name], axis=0)
        sample_size = results[algo_task_name].shape[0]
        std_errors[algo_task_name] = standard_error(std, sample_size, 0.95)

    algonames, _ = zip(*algo_task_names)
    algonames = sorted(set(algonames))

    # 3. Plots
    task_plot(
        xs,
        mus,
        std_errors,
        title,
        Path.cwd()
        / "plots"
        / "debug"
        / "-".join(algonames)
        / title.split(':')[0].upper(),
    )

if __name__ == "__main__":
    # ENV = 'mpe:SimpleTag-v0'
    # ENV = 'rware-tiny-4ag-v1'
    ENV = 'lbforaging:Foraging-15x15-3p-5f-v1'
    # ENV = 'lbforaging:Foraging-15x15-4p-5f-v1'
    algosnames = ('iql_ns', 'ntwql', 'vdn_ns')
    # algosnames = ('ia2c_ns', 'ntwa2c', 'maa2c_ns')
    # source = 'remote'
    # sources = [_q.pop('source') if 'source' in _q else 'remote' for _q in NTWQL_QUERIES[ENV].values()]

    main(ENV, algosnames)
