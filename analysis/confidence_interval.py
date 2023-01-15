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

from utils import standard_error
import src.mongo_db as mdb

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
            'query_ids': [*range(387, 391 + 1)],
            'query_config': {
                'config.name': 'ntwa2c',
                'config.networked_edges': 1,
                'config.networked_rounds': 5,
                'config.networked_interval': 10,
                'config.t_max': 20050000
            }
        },
        'maa2c_ns': {
            'query_ids': [*range(377, 381 + 1)],
            'query_config': {
                'config.name': 'maa2c_ns',
                'config.t_max': 20050000
            }
        },
    }),
    'lbforaging:Foraging-15x15-3p-5f-v1': OrderedDict({
        'ia2c_ns': {
            'query_ids': [*range(402, 406 + 1)],
            'source': 'remote',
            'query_config': {
                'config.name': 'ia2c_ns',
            }
        },
        'ntwa2c': {
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
            'query_ids': [*range(372, 376 + 1)],
            'source': 'remote',
            'query_config': {
                'config.name': 'maa2c_ns',
                'config.t_max': 40050000
            }
        },
    }),
    'lbforaging:Foraging-15x15-4p-5f-v1': OrderedDict({
        'ia2c_ns': { # LONG RUN
            'source': 'remote',
            'query_ids': [*range(359, 363 + 1)],
            'query_config': {
                'config.name': 'ia2c_ns',
                'config.t_max': 40050000
            }
        },
        'ntwa2c': { # LONG RUN
            'source': 'remote',
            'query_ids': [*range(339, 343 + 1)],
            'query_config': {
                'config.name': 'ntwa2c',
                'config.networked_edges': 1,
                'config.networked_rounds': 10,
                'config.networked_interval': 5,
                'config.t_max': 40050000
            }
        },
        'maa2c_ns': { # LONG RUN
            'source': 'remote',
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
            'query_ids': [*range(367, 371 + 1)],
            'source': 'remote',
            'query_config': {
                'config.name': 'iql_ns',
                'config.t_max': 5050000
            }
        },
        'ntwql': OrderedDict({
            'query_ids': [*range(354, 358 + 1)],
            'source': 'remote',
            'query_config': {
                'config.name': 'ntwql',
                'config.networked_edges': 1,
                'config.networked_rounds': 10,
                'config.networked_interval': 5,
                'config.t_max': 5050000
                
            }
        }),
        'vdn_ns': {
            'query_ids': [*range(344, 348 + 1)],
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
            'source': 'remote',
            'query_ids': [226, 227, 228, 229, 351],  # LONG RUN 
            'query_config': {
                'config.name': 'iql_ns',
                'config.t_max': 10_050_000
            }
        },
        'ntwql': {
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

def main(
    environment: str,
    algonames: List[str],
    sources: Union[str, List[str]],
    queries: List[Dict],
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
    """
    # Normalize inputs
    if (len(queries) == 0):
        queries = [{} for _ in range(len(algonames))]
    assert len(queries) == len(algonames)

    if isinstance(sources, str):
        sources = [sources] * len(algonames)
    assert len(sources) == len(algonames)
    assert all([source in ('local', 'remote', 'filesystem') for source in sources])

    # 1. Queries algos and aggregates runs
    for algo, source, query in zip(algonames, sources, queries):

        if source == 'filesystem':
            _, res = file_processor(environment, algo, query)
        elif source in ('remote', 'local'):
            _, res = mongo_processor(environment, algo, source, query)
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
    # ENV = 'mpe:SimpleTag-v0'
    ENV = 'rware-tiny-4ag-v1'
    # ENV = 'lbforaging:Foraging-15x15-3p-5f-v1'
    # ENV = 'lbforaging:Foraging-15x15-4p-5f-v1'
    algonames = list(NTWQL_QUERIES[ENV].keys())
    sources = [_q.pop('source') if 'source' in _q else 'remote' for _q in NTWQL_QUERIES[ENV].values()]
    queries = list(NTWQL_QUERIES[ENV].values())

    main(ENV, algonames, sources, queries)
