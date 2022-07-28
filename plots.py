"""Plots

Reference
---------
[1] Papoudakis, Georgios and Christianos, Filippos and Sch\"{a}fer, Lukas and Albrecht, Stefano 
"Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks",
Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021
"""
import json
from typing import Dict
from pathlib import Path
import re
import string
from collections import defaultdict

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
    std_errors: Array,
    suptitle: str,
    algoname: str,
    save_directory_path: Path = None,
) -> None:
    """Plots Figure 11 for a sigle algo from [1]

    Episodic returns of all algorithms with parameter sharing in all environments
    showing the mean and the 95% confidence interval over five different seeds.

    Parameters
    ----------
    timesteps: Array
        The number of timesteps.
    returns: Array
        The returns collected during training, e.g, rewards.
    std_errors: Array,
        The confidence interval.
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
    normalize_y_axis = "Foraging" in suptitle

    if algoname.startswith("IA2C"):
        marker, color = "x", "C1"
    if algoname.startswith("IPPO"):
        marker, color = "|", "C2"
    elif algoname.startswith("MAA2C"):
        marker, color = "p", "C5"
    elif algoname.startswith("MAPPO"):
        marker, color = "h", "C6"
    elif algoname.startswith("SGLA2C"):
        marker, color = "p", "C9"
    elif algoname.startswith("DSTA2C"):
        marker, color = "*", "C10"
    elif algoname.startswith("INDA2C"):
        marker, color = ">", "C11"
    else:
        marker, color = "^", "C0"

    plt.plot(timesteps, returns, label=algoname, marker=marker, linestyle="-", c=color)
    plt.fill_between(
        timesteps,
        returns - std_errors,
        returns + std_errors,
        facecolor=color,
        alpha=0.25,
    )

    plt.xlabel("Environment Timesteps")
    plt.ylabel("Episodic Return")
    plt.legend(loc=4)
    if normalize_y_axis:
        plt.ylim(bottom=0, top=1.1)
    plt.suptitle(suptitle)
    plt.grid(which="major", axis="both")
    _savefig(suptitle, save_directory_path)
    plt.show()
    plt.close(fig)


def task_plot2(
    timesteps: Dict,
    returns: Dict,
    std_errors: Dict,
    suptitle: str,
    save_directory_path: Path = None,
) -> None:
    """Plots Figure 11 from [1] for many algorithms

    Episodic returns of all algorithms with parameter sharing in all environments
    showing the mean and the 95% confidence interval over five different seeds.


    Parameters
    ----------
    timesteps: Dict[Array]
        Key is the algoname and value is the number of timesteps.
    returns:  Dict[Array]
        Key is the algoname and value is the return collected during training, e.g, rewards.
    std_errors: Dict[Array]
        Key is the algoname and value is the confidence interval.
    suptitle: str,
        The superior title
    ylabel: str
        The name of the metric.
    save_directory_path: Path = None
        Saves the reward plot on a pre-defined path.

    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    normalize_y_axis = "Foraging" in suptitle
    minor_x_ticks = "rware" in suptitle

    for algoname in timesteps:

        if algoname.startswith("IA2C"):
            marker, color = "x", "C1"
        elif algoname.startswith("IPPO"):
            marker, color = "|", "C2"
        elif algoname.startswith("MAA2C"):
            marker, color = "p", "C5"
        elif algoname.startswith("MAPPO"):
            marker, color = "h", "C7"
        elif algoname.startswith("SGLA2C"):
            marker, color = "p", "C3"
        elif algoname.startswith("DSTA2C"):
            marker, color = "*", "C4"
        elif algoname.startswith("INDA2C"):
            marker, color = ">", "C6"
        else:
            marker, color = "^", "C0"
        X = timesteps[algoname]
        Y = returns[algoname]
        err = std_errors[algoname]
        plt.plot(X, Y, label=algoname, marker=marker, linestyle="-", c=color)
        plt.fill_between(X, Y - err, Y + err, facecolor=color, alpha=0.25)

    plt.xlabel("Environment Timesteps")
    plt.ylabel("Episodic Return")
    plt.legend(loc=4)
    if normalize_y_axis:
        plt.ylim(bottom=0, top=1.1)
    plt.suptitle(suptitle)
    if minor_x_ticks:
        x_ticks = [x for x in X if (x - 5_000) % 5_000_000 == 0]
        plt.xticks(ticks=x_ticks)

    plt.grid(which="major", axis="both")
    _savefig(suptitle, save_directory_path)
    _savefig(suptitle, save_directory_path)
    plt.show()
    plt.close(fig)


def main():
    """Plots many models for the same task"""
    BASE_PATH = Path("results/sacred/")
    # algos_paths = BASE_PATH.glob("*a2c")  # Pattern matching ia2c and maa2c
    # algos_paths = BASE_PATH.glob("maa2c_ns")  # Only look for maa2c_ns
    # Match many algorithms
    algos_paths = []
    for pattern in ("maa2c", "sgla2c", "dsta2c"):
        algos_paths += [*BASE_PATH.glob(pattern)]

    def task_matcher(x):
        _paths = []
        for _pattern in (
            "Foraging-8x8-2p-2f-coop*",
            "Foraging-10x10-3p-3f*",
            "Foraging-15x15-3p-5f*",
            "Foraging-15x15-4p-3f*",
        ):
            _paths += [*x.glob(f"lbforaging:{_pattern}")]
        return _paths

    steps = defaultdict(list)
    results = defaultdict(list)
    for algo_path in algos_paths:
        algo_name = algo_path.stem.upper()
        # Matches every lbforaging task.
        # for task_path in algo_path.glob("lbforaging*"):
        for task_path in task_matcher(algo_path):

            task_name = task_path.stem.split(":")[-1].split("-v")[0]
            sample_size = 0

            for path in task_path.rglob("metrics.json"):
                with path.open("r") as f:
                    data = json.load(f)

                if "test_return_mean" in data:
                    _steps = data["test_return_mean"]["steps"]
                    _values = data["test_return_mean"]["values"]
                    print(f"source: {task_name} n_points:{len(_values)}")

                    # Get at most the 41 first evaluations
                    steps[(algo_name, task_name)].append(_steps[:41])
                    results[(algo_name, task_name)].append(_values[:41])
                    sample_size += 1

            if sample_size > 0:
                steps[(algo_name, task_name)] = np.vstack(steps[(algo_name, task_name)])
                results[(algo_name, task_name)] = np.vstack(
                    results[(algo_name, task_name)]
                )

    # Unique algos and tasks
    algo_names, task_names = zip(*[*results.keys()])
    algo_names, task_names = sorted(set(algo_names)), sorted(set(task_names))

    # Makes a plot per task
    for task_name in task_names:
        xs = defaultdict(list)
        mus = defaultdict(list)
        std_errors = defaultdict(list)

        for algo_name in algo_names:
            if (algo_name, task_name) in steps:
                # Computes average returns
                xs[algo_name] = np.mean(steps[(algo_name, task_name)], axis=0)
                mus[algo_name] = np.mean(results[(algo_name, task_name)], axis=0)
                std = np.std(results[(algo_name, task_name)], axis=0)
                std_errors[algo_name] = standard_error(std, sample_size, 0.95)

        task_plot2(
            xs,
            mus,
            std_errors,
            task_name,
            Path.cwd()
            / "plots"
            / "-".join(algo_names)
            / task_name.split("-")[0].upper(),
        )


if __name__ == "__main__":
    main()
