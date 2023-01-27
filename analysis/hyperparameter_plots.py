"""Plots
Reference
---------
[1] Papoudakis, Georgios and Christianos, Filippos and Sch\"{a}fer, Lukas and Albrecht, Stefano
"Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks",
Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021
"""
from pathlib import Path
from operator import itemgetter
from collections import defaultdict

from src.utils.stats import standard_error
from src.utils.loaders import loader
from src.utils.plots import task_plot

import numpy as np
import pandas as pd
Array = np.ndarray

def main(
    environment: str,
    algoname: str,
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
    algoname:  str
        A N_ALGO sized list of strings each of which matching C Name
        examples: ['ia2c_ns', 'ntwa2c'] or ['iql_ns', 'ntwql']
    suptitle: str, default ''
        The superior title's subtitle
    """

    title = ''
    df = loader(environment, algoname, hypergroup=True)

    # Iteration columns
    environments = df.columns.get_level_values(0)
    hypergroups = df.columns.get_level_values(1)
    keys = {eh for eh in zip(environments, hypergroups)}
    keys = sorted(sorted(keys, key=itemgetter(1)), key=itemgetter(0))

    for key in keys:
        title = f"{suptitle} {key[-1]}"

        query = [(*key, i) for i in range(3)]

        hp = pd.concat([df.xs(q, level=(0, 1, 2), axis=1) for q in query], axis=1)

        # 2. Unite runs and generate group statistics
        algo_task_name = (algoname.upper(), environment)

        # Makes a plot per task
        xs = defaultdict(list)
        mus = defaultdict(list)
        std_errors = defaultdict(list)
        xs[algo_task_name] = hp.index
        values = hp.xs(key, level=(0, 1), axis=1)
        mus[algo_task_name] = values.mean(axis=1)
        std = values.std(axis=1)
        std_errors[algo_task_name] = standard_error(std, values.shape[1], 0.95)

        # 3. Plots
        task_plot(
            xs,
            mus,
            std_errors,
            title,
            Path.cwd()
            / "plots"
            / "dataframes"
            / algoname
            / title.split(':')[0].upper(),
        )


if __name__ == "__main__":
    # ENV = 'mpe:SimpleTag-v0'
    # ENV = 'rware:rware-tiny-4ag-v1'
    ENV = 'lbforaging:Foraging-15x15-3p-5f-v1'

    algoname = 'ntwql'
    suptitle = 'TestHyperparameterGroup'
    main(ENV, algoname, suptitle)
