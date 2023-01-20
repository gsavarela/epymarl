"""Plots
Reference
---------
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
import matplotlib.pyplot as plt
# legends for multiple x-axis
import matplotlib.lines as mlines
import matplotlib
matplotlib.use('QtCairo')

Array = np.ndarray
FIGURE_X = 6.0
FIGURE_Y = 4.0
MEAN_CURVE_COLOR = (0.89, 0.282, 0.192)
SMOOTHING_CURVE_COLOR = (0.33, 0.33, 0.33)
SEED_PATTERN = r"seed=(.*?)\)"
M_PATTERN = r"M=(.*?)\,"


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

def task_plot(
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
    # minor_x_ticks = "rware" in suptitle
    # normalize_y_axis = False
    minor_x_ticks = False

    for algo_task_name in timesteps:
        algoname, taskname = algo_task_name


        if algoname.startswith("NTW"):
            marker, color = "^", "C0"
        elif algoname.startswith("IA2C") or algoname.startswith("IQL"):
            marker, color = "x", "C1"
        elif algoname.startswith("MAPPO"):
            marker, color = "h", "C7"
        elif algoname.startswith("MAA2C") or algoname.startswith("VDN"):
            marker, color = "p", "C5"
        else:
            raise ValueError(f'{algoname} not recognizable.')
        X = timesteps[algo_task_name]
        Y = returns[algo_task_name]
        err = std_errors[algo_task_name]
        plt.plot(X, Y, label=ALGO_ID_TO_ALGO_LBL[algoname], marker=marker, linestyle="-", c=color)
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

def main(
    environment: str,
    algonames: List[str],
    sources: Union[str, List[str]],
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
    sources: Union[str, List[str]]
        Either a N_ALGO sized list of strings or string.
        examples: 'local' or ['remote', 'local']
        choice:
    suptitle: str, default ''
        The superior title's subtitle
    """
    if isinstance(sources, str):
        sources = [sources] * len(algonames)
    assert len(sources) == len(algonames)
    assert all([source in ('local', 'remote', 'filesystem') for source in sources])


    title = ''
    steps = defaultdict(list)
    results = defaultdict(list)

    # 1. Queries algos and aggregates runs
    for algo, source in zip(algonames, sources):

        if source == 'filesystem':
            raise ValueError()
            # _steps, _results = file_processor(environment, algo, query)
        elif source in ('remote', 'local'):
            _steps, _results = loader(environment, algo, source)
        else:
            raise ValueError()
            
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
        / "supplemental"
        / "-".join(algonames)
        / title.split(':')[0].upper(),
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
    ENV = 'mpe:SimpleTag-v0'
    # ENV = 'rware-tiny-4ag-v1'
    # ENV = 'lbforaging:Foraging-15x15-3p-5f-v1'
    # ENV = 'lbforaging:Foraging-15x15-4p-5f-v1'
    # algosnames = ('iql_ns', 'ntwql', 'vdn_ns')
    algosnames = ('ia2c_ns', 'ntwa2c', 'maa2c_ns')
    source = 'remote'
    # sources = [_q.pop('source') if 'source' in _q else 'remote' for _q in NTWQL_QUERIES[ENV].values()]

    main(ENV, algosnames, source)

