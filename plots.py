"""Plots

Reference
---------
[1] Papoudakis, Georgios and Christianos, Filippos and Sch\"{a}fer, Lukas and Albrecht, Stefano 
"Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks",
Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021
"""
import json
from typing import List
from pathlib import Path
import re
import string

from utils import standard_error
import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray
FIGURE_X = 6.0
FIGURE_Y = 4.0
MEAN_CURVE_COLOR = (0.89, 0.282, 0.192)
SMOOTHING_CURVE_COLOR = (0.33, 0.33, 0.33)
SEED_PATTERN = r"seed=(.*?)\)"
M_PATTERN = r"M=(.*?)\,"

def _savefig(suptitle: str, save_directory_path: Path = None) -> None:
    """Saves a figure, named after suptitle, if save_directory_path is provided

    Parameters:
    ----------
    suptitle: str
        The title.
    save_directory_path: Path = None
        Saves the reward plot on a pre-defined path.

    Returns:
    -------
    filename: str
        Space between words are filled with underscore (_).
    """
    if save_directory_path is not None:
        # iteractively check for path and builds where it doesnt exist.
        prev_path = None
        for sub_dir in save_directory_path.parts:
            if prev_path is None:
                prev_path = Path(sub_dir)
            else:
                prev_path = prev_path / sub_dir
            prev_path.mkdir(exist_ok=True)
        # uses directory
        file_path = save_directory_path / "{0}.png".format(_to_filename(suptitle))
        plt.savefig(file_path.as_posix())


def _to_filename(suptitle: str) -> str:
    """Formats a plot title to filenme"""

    # Tries to search for a seed pattern and than
    # a M_PATTERN
    gname = "seed"
    match = re.search(SEED_PATTERN, suptitle)
    if match is None:
        gname = "pipeline"
        match = re.search(M_PATTERN, suptitle)
        if match is None:
            return _snakefy(suptitle)
    preffix = suptitle[: (min(match.span()) - 2)]
    group = match.group(1)
    filename = "{0}-{1}{2:02d}".format(_snakefy(preffix), gname, int(group))
    return filename


def _snakefy(title_case: str) -> str:
    """Converts `Title Case` into `snake_case`

    Parameters
    ----------
    title_case: str
        Uppercase for new words and spaces to split then up.

    Returns
    -------
    filename: str
        Space between words are filled with underscore (_).
    """
    fmt = title_case.translate(str.maketrans("", "", string.punctuation))
    return "_".join(fmt.lower().split())

def task_plot(
    timesteps: Array,
    returns: Array,
    c_is: Array,
    suptitle: str,
    algoname: str,
    save_directory_path: Path = None,
) -> None:
    """Plots Figure 11 from [1]

    Episodic returns of all algorithms with parameter sharing in all environments
    showing the mean and the 95% confidence interval over five different seeds.

    Parameters
    ----------
    timesteps: Array
        The number of timesteps.
    returns: Array
        The returns collected during training, e.g, rewards.
    c_is: Array,
        The confidence interval.
    suptitle: str,
    algoname: str,
        Legend
    ylabel: str
        The name of the metric.
    suptitle: str
        The title.
    save_directory_path: Path = None
        Saves the reward plot on a pre-defined path.

    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    # TODO: Place style for different algos.
    # e.g, IA2C -- orange and 'x'
    plt.plot(timesteps, returns, label=algoname, marker='x', linestyle='-', c='C1')
    plt.fill_between(timesteps, returns - c_is, returns + c_is, facecolor="C1", alpha=0.25)

    plt.xlabel("Environment Timesteps")
    plt.ylabel("Episodic Return")
    plt.legend(loc=4)
    plt.ylim(bottom=0, top=1.1)
    plt.suptitle(suptitle)
    plt.grid(which='major', axis='both')
    _savefig(suptitle, save_directory_path)
    plt.show()
    plt.close(fig)

BASE_PATH = Path('results/sacred/ia2c/')
for base_path in BASE_PATH.glob('lbforaging*'):
    test_returns = []
    test_steps = []
    sample_size = 0
    for path in base_path.rglob('metrics.json'):
        with path.open('r') as f:
            data = json.load(f)

        if 'test_return_mean' in data:
            values = data['test_return_mean']['values']
            print(f'source: {path.parent} n_points:{len(values)}')
            test_returns.append(data['test_return_mean']['values'])
            test_steps.append(data['test_return_mean']['steps'])
            sample_size += 1

    X = np.vstack(test_steps)
    Y = np.vstack(test_returns)

    # Computes average returns
    mu_y = np.mean(Y, axis=0)
    std_y = np.std(Y, axis=0)
    ci_y = standard_error(std_y, sample_size, 0.95)
    taskname = base_path.stem.split(':')[-1]

    task_plot(np.mean(X, axis=0), mu_y, ci_y, taskname, 'IA2C', Path.cwd())
