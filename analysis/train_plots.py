from pathlib import Path
from operator import itemgetter
from collections import defaultdict
from typing import List

from src.utils.stats import standard_error
from src.utils.loaders import loader
from src.utils.plots import task_plot

import numpy as np
import pandas as pd

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
    algo_task_names = []

    # 1. Queries algos and aggregates runs
    dataframes = []
    for algo in algonames:
        df = loader(environment, algo, hypergroup=False)
        algo_task_names.append((algo.upper(), environment))
        dataframes.append(df)
            
        taskname = environment
        title = taskname
        if len(suptitle) > 1:
            title = f"{taskname} ({suptitle})"
        
    # Concatenates by average step
    # TODO: Find a better way to collapse indexes
    index = np.mean(np.vstack([df.index for df in dataframes]), axis=0).astype(int)
    for df in dataframes:
        df.set_index(index, inplace=True)
    df = pd.concat(dataframes, axis=1)
    # 2. Unite runs and generate group statistics
    algo_task_names = sorted(sorted(algo_task_names, key=itemgetter(1)), key=itemgetter(0))

    # Makes a plot per task
    xs = defaultdict(list)
    mus = defaultdict(list)
    std_errors = defaultdict(list)
    for algo_task_name in algo_task_names:
        # Computes average returns

        xs[algo_task_name] = df.index
        values = df.xs(algo_task_name[-1::-1], level=(0, 1), axis=1)
        mus[algo_task_name] = values.mean(axis=1)
        std = values.std(axis=1)
        std_errors[algo_task_name] = standard_error(std, values.shape[1], 0.95)

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
        / "dataframes"
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
