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
from operator import itemgetter
from collections import defaultdict
from typing import List

from utils import standard_error
import numpy as np
import matplotlib.pyplot as plt
# legends for multiple x-axis
import matplotlib.lines as mlines

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

def task_plot3(
    timesteps: Dict,
    returns: Dict,
    std_errors: Dict,
    suptitle: str,
    save_directory_path: Path = None,
) -> None:
    """Plots Figure 11 from [1] taking into account observability

    Episodic returns of all algorithms with parameter sharing in all environments
    showing the mean and the 95% confidence interval over five different seeds.

    Parameters
    ----------
    timesteps: Dict[Array]
        Key is the task and value is the number of timesteps.
    returns:  Dict[Array]
        Key is the task and value is the return collected during training, e.g, rewards.
    std_errors: Dict[Array]
        Key is the task and value is the confidence interval.
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

    for algo_task_name in timesteps:

        algoname, taskname = algo_task_name
        ispartial = '-2s-' in taskname
        label = f'{algoname}{"-PO" if ispartial else ""}' 
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
            if ispartial:
                marker, color = ">", "C6"
            else:
                marker, color = "|", "C2"
        else:
            marker, color = "^", "C0"
        X = timesteps[algo_task_name]
        Y = returns[algo_task_name]
        err = std_errors[algo_task_name]
        plt.plot(X, Y, label=label, marker=marker, linestyle="-", c=color)
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
    plt.show()
    plt.close(fig)


def task_plot4(
    timesteps: Dict,
    returns: Dict,
    std_errors: Dict,
    suptitle: str,
    save_directory_path: Path = None,
    dual_x_axis: bool = False
) -> None:
    """Plots Figure 11 from [1] taking into account observability

    Episodic returns of all algorithms with parameter sharing in all environments
    showing the mean and the 95% confidence interval over five different seeds.

    Parameters
    ----------
    timesteps: Dict[Array]
        Key is the task and value is the number of timesteps.
    returns:  Dict[Array]
        Key is the task and value is the return collected during training, e.g, rewards.
    std_errors: Dict[Array]
        Key is the task and value is the confidence interval.
    suptitle: str,
        The superior title
    ylabel: str
        The name of the metric.
    save_directory_path: Path = None
        Saves the reward plot on a pre-defined path.

    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    lines = []
    algonames = []

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
        kwargs = {
            'label': algoname,
            'marker': marker,
            'linestyle': '-',
            'c': color
        }

        
        ax = ax2 if dual_x_axis and algoname.startswith("SGLA2C") else ax1
        X = timesteps[algoname]
        Y = returns[algoname]
        err = std_errors[algoname]
        
        # ax.plot(X, Y, label=algoname, marker=marker, linestyle="-", c=color)
        ax.plot(X, Y, **kwargs)
        ax.fill_between(X, Y - err, Y + err, facecolor=color, alpha=0.25)
        # for dual_x_axis we need to use a proxy artist to explicitly paint
        lines.append(mlines.Line2D([], [], **kwargs))
        algonames.append(algoname)


    ax1.set_xlabel("Environment Timesteps")
    plt.ylabel("Episodic Return")
    ax1.legend(lines, algonames, loc=4)
    # handles, labels = ax1.get_legend_handles_labels()
    if normalize_y_axis:
        plt.ylim(bottom=0, top=1.1)
    plt.suptitle(suptitle)
    if minor_x_ticks:
        x_ticks = [x for x in X if (x - 5_000) % 5_000_000 == 0]
        ax1.set_xticks(ticks=x_ticks)

    if dual_x_axis and 'SGLA2C' in timesteps:
        ax2.set_xlabel("SGLA2C Timesteps")
        ax2.tick_params(axis='x', colors='C3')
        ax2.title.set_color('red')
    ax1.grid(which="major", axis="y")
    _savefig(suptitle, save_directory_path)
    plt.show()
    plt.close(fig)

def ablation_plot(
    timesteps: Dict,
    returns: Dict,
    std_errors: Dict,
    suptitle: str,
    save_directory_path: Path = None,
) -> None:
    """Plots a comparison between parameters

    Parameters
    ----------
    timesteps: Dict[Array]
        Key is the task and value is the number of timesteps.
    returns:  Dict[Array]
        Key is the task and value is the return collected during training, e.g, rewards.
    std_errors: Dict[Array]
        Key is the task and value is the confidence interval.
    suptitle: str,
        The superior title
    ylabel: str
        The name of the metric.
    save_directory_path: Path = None
        Saves the reward plot on a pre-defined path.

    Reference:
    ---------
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    ax = fig.add_subplot(111)
    lines = []
    categorynames = []

    normalize_y_axis = "Foraging" in suptitle
    categorynames = filter(lambda x: x in ('6', '3', '0'), sorted(timesteps.keys(), reverse=True))
    for categoryname in categorynames:
        if categoryname in ('6',):
            linestyle, color = 'solid', 'C1'
        elif categoryname in ('5',):
            linestyle, color = (0, (3, 10, 1, 10, 1, 10)), 'C2' # loosely dashdotted 
        elif categoryname in ('4',):
            linestyle, color = (0, (3, 1, 1, 1, 1, 1)),  'C5' # densily dashdotted
        elif categoryname in ('3',):
            linestyle, color = 'dashdot', 'C7'
        elif categoryname in ('2',):
            linestyle, color = (5, (10, 3)), 'C3' # long dash with offset
        elif categoryname in ('1',):
            linestyle, color = (0, (5, 5)), 'C4' # dashed
        elif categoryname in ('0',):
            linestyle, color = 'dotted', 'C6'
        else:
            linestyle, color = (0, (1, 10)), 'C0' # loosely dotted
        
        X = timesteps[categoryname]
        Y = returns[categoryname]
        err = std_errors[categoryname]
        
        ax.plot(X, Y, label=categoryname, linestyle=linestyle, c=color)
        ax.fill_between(X, Y - err, Y + err, facecolor=color, alpha=0.25)
        # for dual_x_axis we need to use a proxy artist to explicitly paint
        # lines.append(mlines.Line2D([], []))
        # categorynames.append(categoryname)


    ax.set_xlabel("Environment Timesteps")
    plt.ylabel("Episodic Return")
    # ax.legend(lines, categorynames, loc=4)
    ax.legend( loc=4)
    # handles, labels = ax1.get_legend_handles_labels()
    if normalize_y_axis:
        plt.ylim(bottom=0, top=1.1)
    plt.suptitle(suptitle)

    ax.grid(which="major", axis="y")
    _savefig(suptitle, save_directory_path)
    plt.show()
    plt.close(fig)

def main2():
    """Plots aggregating models by  task"""
    BASE_PATH = Path("results/sacred/")
    # algos_paths = BASE_PATH.glob("*a2c")  # Pattern matching ia2c and maa2c
    # algos_paths = BASE_PATH.glob("maa2c_ns")  # Only look for maa2c_ns
    # Match many algorithms
    algos_paths = []
    # for pattern in ("maa2c", "sgla2c", "dsta2c"):
    # for pattern in ("inda2c",):
    for pattern in ("sgla2c",):
        algos_paths += [*BASE_PATH.glob(pattern)]

    def task_matcher(x):
        _paths = []

        for _pattern in (
            # "Foraging-*8x8-2p-2f-coop*",
            # "Foraging-*10x10-3p-3f*",
            # "Foraging-*15x15-3p-5f*",
            # "Foraging-*15x15-4p-3f*",
            "Foraging-*15x15-4p-5f*",
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

def main3(
    algonames: str = ["inda2c"],
    size: int = 8,
    players: int = 2,
    food: int = 2,
    coop: bool = True,
    po: bool = False
):
    """Plots aggregating models by task pattern

    Use to compare partial observable settings and fully
    observable settings.

    po: bool = False
    Force partial observability
    """
    BASE_PATH = Path("results/sacred/")
    # algos_paths = BASE_PATH.glob("*a2c")  # Pattern matching ia2c and maa2c
    # algos_paths = BASE_PATH.glob("maa2c_ns")  # Only look for maa2c_ns
    # Match many algorithms
    algos_paths = []
    for algoname in algonames:
        algos_paths += [*BASE_PATH.glob(algoname)]
    _coop = '-coop' if coop else ''
    title = f'Foraging {size}x{size}-{players}p-{food}f{_coop}'

    def task_pattern_builder(x):
        _paths = []
        partial = '2s-'if po else '*'

        _pattern = f'Foraging-{partial}{size}x{size}-{players}p-{food}f{_coop}'
        _paths += [*x.glob(f"lbforaging:{_pattern}-v2")]
        return _paths

    steps = defaultdict(list)
    results = defaultdict(list)
    for algo_path in algos_paths:
        # Matches every lbforaging task.
        algoname = algo_path.stem.upper()
        print(algoname, task_pattern_builder(algo_path))
        for task_path in task_pattern_builder(algo_path):

            task_name = task_path.stem.split(":")[-1].split("-v")[0]
            key = (algoname, task_name)
            sample_size = 0

            for path in task_path.rglob("metrics.json"):
                with path.open("r") as f:
                    data = json.load(f)

                if "test_return_mean" in data:
                    _steps = data["test_return_mean"]["steps"]
                    _values = data["test_return_mean"]["values"]
                    print(f"algoname: {algoname} source: {task_name} n_points:{len(_values)}")

                    # Get at most the 41 first evaluations
                    steps[key].append(_steps[:41])
                    results[key].append(_values[:41])
                    sample_size += 1

            if sample_size > 0:
                steps[key] = np.vstack(steps[key])
                results[key] = np.vstack(
                    results[key]
                )

    # Unique algos and tasks
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
        std_errors[algo_task_name] = standard_error(std, sample_size, 0.95)

    algonames, _ = zip(*algo_task_names)
    algonames = sorted(set(algonames))
    task_plot3(
        xs,
        mus,
        std_errors,
        title,
        Path.cwd()
        / "plots"
        / "-".join(algonames)
        / title.split()[0].upper(),
    )


def main4(
    algonames: List[str] = ["sgla2c", "dsta2c", "inda2c"],
    size: int = 8,
    players: int = 2,
    food: int = 2,
    coop: bool = False,
    dual_x_axis: bool = False
):
    """Plots multi-models using per model task pattern

    Requires every time series to have the same number of points
    * inda2c: Use partial observable setting.
    * sgla2c: Adjust timestep scale.
    """
    BASE_PATH = Path("results/sacred/")
    # Match many algorithms
    algos_paths = []
    for algoname in algonames:
        algos_paths += [*BASE_PATH.glob(algoname)]
    _coop = '-coop' if coop else ''
    title = f'Foraging {size}x{size}-{players}p-{food}f{_coop}'

    def task_pattern_builder(x):
        _paths = []
        _partial = '-2s' if 'inda2c' in x.as_posix() else ''
        _pattern = f'Foraging{_partial}-{size}x{size}-{players}p-{food}f{_coop}'
        _paths += [*x.glob(f"lbforaging:{_pattern}*")]
        return _paths

    steps = defaultdict(list)
    results = defaultdict(list)
    taskname = title.lower()
    req_ntest = 41  # Required number of tests 
    for algo_path in algos_paths:
        algoname = algo_path.stem.upper()
        # Matches every lbforaging task.
        print(algoname, task_pattern_builder(algo_path))
        for task_path in task_pattern_builder(algo_path):

            sample_size = 0
            for path in task_path.rglob("metrics.json"):
                with path.open("r") as f:
                    data = json.load(f)

                if "test_return_mean" in data:
                    _steps = data["test_return_mean"]["steps"]
                    _values = data["test_return_mean"]["values"]
                    print(f"algo: {algoname}\tsource: {taskname}\tn_points:{len(_values)}")

                    # Get at most the 41 first evaluations
                    steps[algoname].append(_steps[:req_ntest])
                    results[algoname].append(_values[:req_ntest])
                    sample_size += 1

            if sample_size > 0:
                steps[algoname] = np.vstack(steps[algoname])
                results[algoname] = np.vstack(
                    results[algoname]
                )

            n_seeds, n_steps = steps[algoname].shape
            if n_steps < req_ntest:

                print(f'Warning: {algoname} has less points ({n_steps}) than required({req_ntest}). \n' +
                       'Completing series with last observation')
                # If there is not the required number of tests
                prev_step = steps[algoname][:, -2][:, None]
                for i in range(n_steps, req_ntest):
                    step_size = steps[algoname][:, -1][:, None] - prev_step
                    steps[algoname] = np.append(steps[algoname], steps[algoname][:, -1][:, None] + step_size, axis=1)
                    results[algoname] = np.append(results[algoname], results[algoname][:, -1][:, None], axis=1)
                    prev_step = steps[algoname][-1]
                    
    # Makes a plot per task
    xs = defaultdict(list)
    mus = defaultdict(list)
    std_errors = defaultdict(list)
    for algoname in algonames:
        # Computes average returns
        algoname = algoname.upper()
        xs[algoname] = np.mean(steps[algoname], axis=0)
        mus[algoname] = np.mean(results[algoname], axis=0)
        std = np.std(results[algoname], axis=0)
        std_errors[algoname] = standard_error(std, sample_size, 0.95)


    task_plot4(
        xs,
        mus,
        std_errors,
        title,
        Path.cwd()
        / "plots"
        / "-".join(algonames)
        / title.split()[0].upper(),
        dual_x_axis=dual_x_axis
    )

def ablation(
    algoname: str = 'ntwa2c',
    size: int = 15,
    players: int = 4,
    food: int = 5,
    coop: bool = False,
    dual_x_axis: bool = False
):
    """Plots ablation for a particular scenario.

    Uses parent folder as category for comparison.
    """
    BASE_PATH = Path(f"results/sacred/{algoname}/ablation-target")

    # Match many algorithms
    _coop = '-coop' if coop else ''
    title = f'Foraging {size}x{size}-{players}p-{food}f{_coop}'

    def task_pattern_builder(x):
        _paths = []
        _partial = '-2s' if 'inda2c' in x.as_posix() else ''
        _pattern = f'Foraging{_partial}-{size}x{size}-{players}p-{food}f{_coop}'
        _paths += [*x.rglob(f"lbforaging:{_pattern}*")]
        return _paths

    steps = defaultdict(list)
    results = defaultdict(list)
    taskname = title.lower()
    req_ntest = 41  # Required number of tests 

    ablations = sorted(task_pattern_builder(BASE_PATH), key=lambda x: x.parent.stem)
    # Matches every lbforaging task.
    print(algoname, task_pattern_builder(BASE_PATH))
    categorynames = []
    for task_path in ablations:
        categoryname = task_path.parent.stem
        sample_size = 0
        for path in task_path.rglob("metrics.json"):
            with path.open("r") as f:
                data = json.load(f)

            if "test_return_mean" in data:
                _steps = data["test_return_mean"]["steps"]
                _values = data["test_return_mean"]["values"]
                print(f"algo: {algoname}\tsource: {categoryname}\tn_points:{len(_values)}")

                # Get at most the 41 first evaluations
                steps[categoryname].append(_steps[:req_ntest])
                results[categoryname].append(_values[:req_ntest])
                sample_size += 1

        if sample_size > 0:
            steps[categoryname] = np.vstack(steps[categoryname])
            results[categoryname] = np.vstack(
                results[categoryname]
            )

        n_seeds, n_steps = steps[categoryname].shape
        if n_steps < req_ntest:
            print(f'Warning: {categoryname} has less points ({n_steps}) than required({req_ntest}). \n' +
                   'Completing series with last observation')
            # If there is not the required number of tests
            prev_step = steps[categoryname][:, -2][:, None]
            for i in range(n_steps, req_ntest):
                step_size = steps[categoryname][:, -1][:, None] - prev_step
                steps[categoryname] = np.append(steps[categoryname], steps[categoryname][:, -1][:, None] + step_size, axis=1)
                results[categoryname] = np.append(results[categoryname], results[categoryname][:, -1][:, None], axis=1)
                prev_step = steps[categoryname][-1]
        categorynames.append(categoryname)

    # Makes a plot per task
    xs = defaultdict(list)
    mus = defaultdict(list)
    std_errors = defaultdict(list)
    for categoryname in categorynames:
        # Computes average returns
        categoryname = categoryname.upper()
        xs[categoryname] = np.mean(steps[categoryname], axis=0)
        mus[categoryname] = np.mean(results[categoryname], axis=0)
        std = np.std(results[categoryname], axis=0)
        std_errors[categoryname] = standard_error(std, sample_size, 0.95)
    
    
    ablation_plot(
        xs,
        mus,
        std_errors,
        title,
        Path.cwd()
        / "plots"
        / "-".join(categorynames)
        / title.split()[0].upper(),
    )

if __name__ == "__main__":
    # ablation(
    #     algoname='ntwa2c',
    #     size=15,
    #     players=4,
    #     food=5,
    #     coop=False,
    #     dual_x_axis=False)

    main3(algonames=["inda2c", "ntwa2c"], size=10, players=2, food=2, coop=True, po=False)
    # main3(algonames=["inda2c","ntwa2c"], size=15, players=3, food=3, coop=False, po=True)
    # main3(algonames=["inda2c", "ntwa2c"], size=10, players=2, food=3, coop=True, po=False)
    # main3(algonames=["inda2c"], size=15, players=3, food=3, coop=False, po=True)
    # main4(size=8, players=2, food=2, coop=True, dual_x_axis=True)
    # main4(size=10, players=3, food=3, dual_x_axis=True)
    # main4(size=15, players=3, food=5, dual_x_axis=True)
    # main4(size=15, players=4, food=3, dual_x_axis=True)
    # main4(size=15, players=4, food=5, dual_x_axis=True)
    # main3(algoname="inda2c", size=8, players=2, food=2, coop=True)
    # main3(algoname="inda2c", size=10, players=3, food=3, coop=False)
    # main3(algonames=["inda2c", "ntwa2c", "dsta2c"], size=15, players=3, food=5, coop=False)
    # main3(algonames=["inda2c", "ntwa2c", "dsta2c"], size=15, players=4, food=3, coop=False)
    # main3(algonames=["inda2c", "ntwa2c", "dsta2c"], size=15, players=4, food=5, coop=False)
    # main3(algoname="inda2c", size=15, players=4, food=5, coop=False)
    # algonames = ["sgla2c", "dsta2c", "inda2c"]
    # main4(algonames=algonames, size=15, players=4, food=3, coop=False)
    # main4(algonames=algonames, size=15, players=4, food=5, coop=False)
