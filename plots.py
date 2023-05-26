"""
Reference
---------
[1] Papoudakis, Georgios and Christianos, Filippos and Sch\"{a}fer, Lukas and Albrecht, Stefano
"Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks",
Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021
"""
from typing import Dict, Union, Tuple
from pathlib import Path
from operator import itemgetter
import re
import string

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from src.utils.stats import confidence_interval, confidence_interval_bootstrap
# from src.utils.stats import standard_error

FIGURE_X = 6.0
FIGURE_Y = 4.0
MEAN_CURVE_COLOR = (0.89, 0.282, 0.192)
SMOOTHING_CURVE_COLOR = (0.33, 0.33, 0.33)
SEED_PATTERN = r"seed=(.*?)\)"
M_PATTERN = r"M=(.*?)\,"
FIGS_SAVE_DIR = "plots/dataframes"

# legends for multiple x-axis
# import matplotlib.lines as mlines
matplotlib.use('QtCairo')

Array = np.ndarray

ALGO_ID_TO_ALGO_LBL = {
    'NTWA2C': 'NTWA2C+PC',
    'IA2C_NS': 'IA2C_NS',
    'IA2C': 'IA2C',
    'MAA2C_NS': 'MAA2C_NS',
    'MAA2C': 'MAA2C',
    'NTWQL': 'NTWQL',
    'IQL_NS': 'IQL_NS',
    'VDN_NS': 'VDN_NS',
    'IQL': 'IQL',
    'VDN': 'VDN',
    'INDA2C': 'INDA2C',
    'DACV': 'DACV'
}

def task_plot(
    timesteps: Dict[Tuple[str], Array],
    results: Dict[Tuple[str], Array],
    std_errors: Dict[Tuple[str], Array],
    suptitle: str,
    save_directory_path: Path = None,
) -> None:
    """Plots Figure 11 from [1] taking into account observability

    Episodic results of all algorithms with parameter sharing in all environments
    showing the mean and the 95% confidence interval over five different seeds.

    Parameters
    ----------
    timesteps: Dict[Tuple[str], Array]
        Key is the task and value is the number of timesteps.
    results: Dict[Tuple[str], Array]
        Key is the task and value is the return collected during training, e.g, rewards.
    std_errors: Dict[Tuple[str], Array]
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
        elif algoname.startswith("INDA2C"):
            marker, color = "|", "C3"
        elif algoname.startswith("DACV"):
            marker, color = "h", "C7"
        elif algoname.startswith("MAA2C") or algoname.startswith("VDN"):
            marker, color = "p", "C5"
        else:
            raise ValueError(f'{algoname} not recognizable.')
        X = timesteps[algo_task_name]
        Y = results[algo_task_name]
        err = std_errors[algo_task_name]
        plt.plot(X, Y, label=ALGO_ID_TO_ALGO_LBL[algoname], marker=marker, linestyle="-", c=color)
        plt.fill_between(X, Y - err, Y + err, facecolor=color, alpha=0.25)

    plt.xlabel("Environment Timesteps")
    plt.ylabel("Episodic Return")
    plt.legend(loc='best')
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

################################################################
# Plot unif. performance (bar chart).
################################################################
def barchart(environment: str, algo: str, hp_df: pd.DataFrame, max_returns=True, bootstrap=True):
    """BarChart: Return vs Hyper Parameter Group"""
    fig = plt.figure()
    fig.set_size_inches(7.0, 5.0)
    fig.tight_layout()

    errors = []
    ys = []

    # Iteration columns
    # TODO: Create a generator to iterate on groups.
    environments = hp_df.columns.get_level_values(0)
    hypergroups = hp_df.columns.get_level_values(1)
    keys = {eh for eh in zip(environments, hypergroups)}
    keys = sorted(sorted(keys, key=itemgetter(1)), key=itemgetter(0))

    suffix_a = 'MaxReturns' if max_returns else 'AvgReturns'
    suffix_b = 'Bootstrap' if bootstrap else 'Standard'
    suptitle = f"{environment.split(':')[-1]} ({suffix_a}, {suffix_b})"
    for key in keys:
        columns = [(*key, i) for i in range(3)]

        df = pd.concat([hp_df.xs(q, level=(0, 1, 2), axis=1) for q in columns], axis=1)

        # Here we check the returns
        if max_returns:
            idx = df.values.sum(axis=1).argmax()
            mub = min(idx + 3, df.values.shape[0])     # Upper limit
            mlb = max(idx - 2, 0)   # Lower limit
            mub, mlb = mub + (2 - (idx - mlb)), mlb - (3 - (mub - idx))  # Adjust limits
            
            data = df.values[mlb:mub, :].flatten()

            ylabel = 'Max Returns'
        else:
            data = df.values.flatten()
            ylabel = 'Avg Returns'

        y = data.mean()
        if bootstrap:
            ci = confidence_interval_bootstrap(data)
        else:
            ci = confidence_interval(data, data.shape[0], 0.95)

        # Move this to plots.py
        error_top = ci[1] - y
        error_bottom = y - ci[0]
        errors.append([error_top, error_bottom])
        ys.append(y)
    errors = np.array(errors).T
    xs = np.arange(len(ys))
    plt.errorbar(xs, ys,
        yerr=np.max(errors, 0),
        fmt='o',
        capsize=6,
        zorder=50,
    )
    plt.xticks(xs, labels=xs)

    plt.grid()
    plt.ylabel(ylabel)
    plt.xlabel('Hyperparameter Groups')
    plt.suptitle(suptitle)

    _savefig(
        suptitle,
        Path.cwd()
        / 'plots'
        / 'dataframes'
        / algo.upper()
        / environment.split(':')[0].upper()
    )
    plt.show()

################################################################
# Plot sensitivity (bar chart).
################################################################
def barchart_sensitivity(environment: str, algo: str, sen_df: pd.DataFrame, max_returns=True, bootstrap=True):
    """BarChart: Return vs Hyper Parameter Group"""
    fig = plt.figure()
    fig.set_size_inches(7.0, 5.0)
    fig.tight_layout()

    categories = ['networked_edges', 'networked_rounds', 'networked_interval']

    # Iteration columns
    # TODO: Create a generator to iterate on groups.
    # environments = hp_df.columns.get_level_values(0)
    # hypergroups = hp_df.columns.get_level_values(1)
    # keys = {eh for eh in zip(environments, hypergroups)}
    # keys = sorted(sorted(keys, key=itemgetter(1)), key=itemgetter(0))
    #
    suffix_a = 'MaxReturns' if max_returns else 'AvgReturns'
    suffix_b = 'Bootstrap' if bootstrap else 'Standard'
    # suptitle = f"{environment.split(':')[-1]} ({suffix_a}, {suffix_b})"
    #
    prefix = f"{environment.split(':')[-1]}"
    # for key in keys:
    #     columns = [(*key, i) for i in range(3)]
    #
    #     df = pd.concat([hp_df.xs(q, level=(0, 1, 2), axis=1) for q in columns], axis=1)
    #
    #     # Here we check the returns
    #     if max_returns:
    #         idx = df.values.sum(axis=1).argmax()
    #         mub = min(idx + 3, df.values.shape[0])     # Upper limit
    #         mlb = max(idx - 2, 0)   # Lower limit
    #         mub, mlb = mub + (2 - (idx - mlb)), mlb - (3 - (mub - idx))  # Adjust limits
    #         
    #         data = df.values[mlb:mub, :].flatten()
    #
    #         ylabel = 'Max Returns'
    #     else:
    #         data = df.values.flatten()
    #         ylabel = 'Avg Returns'
    for category in categories:
        suffix_c = category.title().replace('_', '')
        suptitle = f'{prefix} ({suffix_a}, {suffix_b}, {suffix_c})'
        xs = sen_df[category].unique()

        errors = []
        ys = []
        for x in xs:
            data = sen_df.loc[sen_df[category] == x, 'value']
            y = data.mean()
            ylabel = suffix_a
            if bootstrap:
                ci = confidence_interval_bootstrap(data)
            else:
                ci = confidence_interval(data, data.shape[0], 0.95)

            # Move this to plots.py
            error_top = ci[1] - y
            error_bottom = y - ci[0]
            errors.append([error_top, error_bottom])
            ys.append(y)
        errors = np.array(errors).T
        plt.errorbar(xs, ys,
            yerr=np.max(errors, 0),
            fmt='o',
            capsize=6,
            zorder=50,
        )
        plt.xticks(xs, labels=xs)

        plt.grid()
        plt.ylabel(ylabel)
        plt.xlabel(suffix_c)
        plt.suptitle(suptitle)

        _savefig(
            suptitle,
            Path.cwd()
            / 'plots'
            / 'dataframes'
            / algo.upper()
            / environment.split(':')[0].upper()
            / 'sensitivity'
        )
        plt.show()
