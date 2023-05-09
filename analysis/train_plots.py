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

from utils import standard_error
import src.mongo_db as mdb

import numpy as np
import matplotlib.pyplot as plt
# legends for multiple x-axis
import matplotlib.lines as mlines
import matplotlib
matplotlib.use('QtCairo')

from incense import ExperimentLoader

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

NTWA2C_QUERIES = OrderedDict({
    'mpe:SimpleTag-v0': OrderedDict({
        'ia2c_ns': {
            'query_ids': [*range(397, 401 + 1)],
            'query_config': {
                'config.name': 'ia2c_ns',
                'config.t_max': 20050000
            }
        },
        'ntwa2c': {
            # 'query_ids': [*range(244, 248 + 1)],
            'query_ids': [*range(387, 391 + 1)],
            'query_config': {
                'config.name': 'ntwa2c',
                # 'config.networked_edges': 1,
                # 'config.networked_rounds': 1,
                # 'config.networked_interval': 5,
                'config.networked_edges': 1,
                'config.networked_rounds': 5,
                'config.networked_interval': 10,
                'config.t_max': 20050000
            }
        },
        'maa2c_ns': {
            # 'query_ids': [*range(249, 253 + 1)],
            'query_ids': [*range(377, 381 + 1)],
            'query_config': {
                'config.name': 'maa2c_ns',
                'config.t_max': 20050000
            }
        },
    }),
    'lbforaging:Foraging-15x15-3p-5f-v1': OrderedDict({
        'ia2c_ns': {
            # 'source': 'filesystem',
            # 'sub_dir': 'random_seeds',
            # 'query_ids': [*range(22, 26 + 1)],
            'query_ids': [*range(402, 406 + 1)],
            'source': 'remote',
            'query_config': {
                'config.name': 'ia2c_ns',
            }
        },
        'ntwa2c': {
            # 'source': 'filesystem',
            # 'sub_dir': 'random_seeds',
            # 'query_ids': [*range(154, 158 + 1)],
            'query_ids': [*range(392, 396 + 1)],
            'query_config': {
                'config.name': 'ntwa2c',
                'config.networked_edges': 1,
                'config.networked_rounds': 10,
                'config.networked_interval': 5,
                'config.t_max': 40050000
            }
        },
        'maa2c_ns': {
            # 'query_ids': [*range(215, 219 + 1)],
            # 'query_ids': [*range(195, 199 + 1)],
            'query_ids': [*range(372, 376 + 1)],
            'source': 'remote',
            'query_config': {
                'config.name': 'maa2c_ns',
                'config.t_max': 40050000
            }
        },
    }),
    'lbforaging:Foraging-15x15-4p-5f-v1': OrderedDict({
        # 'ia2c_ns': {
        #     'source': 'filesystem',
        #     'sub_dir': '30_000_000',
        #     'query_ids': [*range(6, 10 + 1)],
        #     'query_config': {
        #         'config.name': 'ia2c_ns',
        #     }
        # },
        # 'ntwa2c': {
        #     'source': 'filesystem',
        #     'sub_dir': '30_000_000',
        #     'query_ids': [*range(185, 189 + 1)],
        #     'query_config': {
        #         'config.name': 'ntwa2c',
        #     }
        # },
        # 'maa2c_ns': {
        #     'source': 'remote',
        #     # 'query_ids': [*range(22, 26 + 1)],
        #     'query_ids': [*range(215, 219 + 1)],
        #     'query_config': {
        #         'config.name': 'maa2c_ns',
        #     }
        # },
        'ia2c_ns': { # LONG RUN
            'source': 'remote',
            # 'query_ids': [*range(339, 343 + 1)],
            'query_ids': [*range(359, 363 + 1)],
            'query_config': {
                'config.name': 'ia2c_ns',
                'config.t_max': 40050000
            }
        },
        'ntwa2c': { # LONG RUN
            'source': 'remote',
            # 'query_ids': [*range(339, 343 + 1)],
            'query_ids': [*range(339, 343 + 1)],
            'query_config': {
                'config.name': 'ntwa2c',
                # 'config.networked_edges': 2,
                # 'config.networked_rounds': 10,
                # 'config.networked_interval': 5,
                'config.networked_edges': 1,
                'config.networked_rounds': 10,
                'config.networked_interval': 5,
                'config.t_max': 40050000
            }
        },
        'maa2c_ns': { # LONG RUN
            'source': 'remote',
            # 'query_ids': [*range(22, 26 + 1)],
            'query_ids': [*range(318, 322 + 1)],
            'query_config': {
                'config.name': 'maa2c_ns',
                'config.t_max': 40050000
            }
        },
    })
})

NTWQL_QUERIES = OrderedDict({
    'mpe:SimpleTag-v0': OrderedDict({
        'iql_ns': {
            # 'query_ids': [*range(58, 62 + 1)],
            'query_ids': [*range(367, 371 + 1)],
            # 'source': 'local',
            'source': 'remote',
            'query_config': {
                'config.name': 'iql_ns',
                'config.t_max': 5050000
            }
        },
        'ntwql': OrderedDict({
            # 'query_ids': [*range(63, 67 + 1)],
            # 'source': 'local',
            'query_ids': [*range(354, 358 + 1)],
            'source': 'remote',
            'query_config': {
                'config.name': 'ntwql',
                # 'config.networked_edges': 1,
                # 'config.networked_rounds': 10,
                # 'config.networked_interval': 5,
                'config.t_max': 5050000
                
            }
        }),
        'vdn_ns': {
            # 'query_ids': [*range(68, 72 + 1)],
            'query_ids': [*range(344, 348 + 1)],
            # 'source': 'local',
            'source': 'remote',
            'query_config': {
                'config.name': 'vdn_ns',
                'config.t_max': 5005000
            }
        }
    }),
    'lbforaging:Foraging-15x15-3p-5f-v1': OrderedDict({
        'iql_ns': {
            'query_ids': [*range(210, 214 + 1)],
            'query_config': {
                'config.name': 'iql_ns',
            }
        },
        'ntwql': {
            'query_ids': [*range(205, 209 + 1)],
            'query_config': {
                'config.name': 'ntwql',
                'config.networked_edges': 2,
                'config.networked_rounds': 1,
                'config.networked_interval': 5,
            }
        },
        'vdn_ns': {
            'query_ids': [*range(220, 224 + 1)],
            'query_config': {
                'config.name': 'vdn_ns',
            }
        }
    }),
    'lbforaging:Foraging-15x15-4p-5f-v1': OrderedDict({
        'iql_ns': {
            'query_ids': [*range(83, 87 + 1)],
            'source': 'local',
            'query_config': {
                'config.name': 'iql_ns',
                'config.t_max': 5050000
            }
        },
        'ntwql': {
            'query_ids': [*range(78, 82 + 1)],
            'source': 'local',
            'query_config': {
                'config.name': 'ntwql',
                'config.networked_edges': 2,
                'config.networked_rounds': 1,
                'config.networked_interval': 5,
                'config.t_max': 5050000
            }
        },
        'vdn_ns': {
            'query_ids': [*range(73, 77 + 1)],
            'source': 'local',
            'query_config': {
                'config.name': 'vdn_ns',
                'config.t_max': 5050000


            }
        }
    }),
    'rware-tiny-4ag-v1': OrderedDict({
        'iql_ns': {
            # 'query_ids': [*range(58, 62 + 1)],
            'source': 'remote',
            # 'query_ids': [*range(226, 229 + 1)],  # LONG RUN 
            'query_ids': [226, 227, 228, 229, 351],  # LONG RUN 
            'query_config': {
                'config.name': 'iql_ns',
                'config.t_max': 10_050_000
            }
        },
        'ntwql': {
            # 'query_ids': [*range(63, 67 + 1)],
            'query_ids': [263, 264, 265, 267, 365], # LONG RUN
            'source': 'remote',
            'query_config': {
                'config.name': 'ntwql',
                'config.networked_edges': 2,
                'config.networked_rounds': 1,
                'config.networked_interval': 5,
                'config.t_max': 10_050_000
            }
        },
        'vdn_ns': {
            # 'query_ids': [*range(68, 72 + 1)],
            'query_ids': [313, 314, 315, 316, 366],
            'source': 'remote',
            'query_config': {
                'config.name': 'vdn_ns',
                'config.t_max': 10_050_000
            }
        }
    })
})


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

def mongo_processor(environment: str, algo: str, source: str, query: Dict) -> Tuple[Dict]:

    experiments = mongo_loader(environment, algo, source, query)

    steps, results = mongo_parser(environment, algo, experiments)

    return steps, results


def mongo_loader(environment: str, algo: str, source: str, query: Dict) -> List[object]:
    mongo_uri = mdb.build_conn(source)

    loader = ExperimentLoader(mongo_uri=mongo_uri, db_name=mdb.MONGO_DB_NAME)

    mongo_query = mongo_query_builder(environment, algo, query)

    print(mongo_query)
    experiments = loader.find(mongo_query)

    return experiments

def mongo_parser(environment:str, algo: str, experiments: List[object]) -> Tuple[Dict]:

        steps = defaultdict(list)
        results = defaultdict(list)
        sample_size = 0
        algoname = algo.upper()
        taskname = environment
        max_rollouts = 41
        # max_rollouts = 101
        # title = taskname
        # if len(suptitle) > 1:
        #     title = f"{taskname} ({suptitle})"
        key = (algoname, taskname)
        for experiment in experiments:
            ts = experiment.metrics["test_return_mean"]
            index  =  ts.index.to_list()
            values = ts.values.tolist()
            print(f"algoname: {algoname} source: {taskname} n_points:{len(index[:max_rollouts])}")

            steps[key].append(index[:max_rollouts])
            results[key].append(values[:max_rollouts])
            sample_size += 1

        if sample_size > 0:
            steps[key] = np.vstack(steps[key])
            results[key] = np.vstack(
                results[key]
            )
        return steps, results

def mongo_query_builder(environment: str, algoname: str, query: Dict) -> Dict:
    """ 
    Example:
    --------
    {
        'query_config': {
            'config.env_args.key': ENV,
            'config.name': ALGO_ID,
            'config.networked_edges': 2,
            'config.networked_rounds': 1,
            'config.networked_interval': 1,
        },
        'query_ids': QUERY_IDS,
    }
    """

    if 'query_config' not in query:
        query['query_config'] = {}
    query['query_config']['config.env_args.key'] = environment
    query['query_config']['config.name'] = algoname

    if 'query_ids' in query:
        return {'$and': [
            {'_id': { "$in": query['query_ids']}},
            query["query_config"],
        ]}
    else:
        return query["query_config"]


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

def main(
    environment: str,
    algonames: List[str],
    sources: Union[str, List[str]],
    queries: List[Dict],
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
    queries: List[Dict]
        A list with dicts containg queries for each algoname
    suptitle: str, default ''
        The superior title's subtitle
    """

    # Normalize inputs
    if (len(queries) == 0):
        queries = [{} for _ in range(len(algonames))]
    assert len(queries) == len(algonames)

    if isinstance(sources, str):
        sources = [sources] * len(algonames)
    assert len(sources) == len(algonames)
    assert all([source in ('local', 'remote', 'filesystem') for source in sources])


    title = ''
    steps = defaultdict(list)
    results = defaultdict(list)

    # 1. Queries algos and aggregates runs
    for algo, source, query in zip(algonames, sources, queries):

        if source == 'filesystem':
            _steps, _results = file_processor(environment, algo, query)
        elif source in ('remote', 'local'):
            _steps, _results = mongo_processor(environment, algo, source, query)
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
        / "nonlinear"
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
    # ENV = 'mpe:SimpleTag-v0'
    ENV = 'rware-tiny-4ag-v1'
    # ENV = 'lbforaging:Foraging-15x15-3p-5f-v1'
    # ENV = 'lbforaging:Foraging-15x15-4p-5f-v1'
    algonames = list(NTWQL_QUERIES[ENV].keys())
    sources = [_q.pop('source') if 'source' in _q else 'remote' for _q in NTWQL_QUERIES[ENV].values()]
    queries = list(NTWQL_QUERIES[ENV].values())

    main(ENV, algonames, sources, queries, '')

