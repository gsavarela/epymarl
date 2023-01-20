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
import pandas as pd
from operator import itemgetter
from collections import defaultdict, OrderedDict
import scipy.stats as stats
from typing import List

from analysis.stats import standard_error
from src.utils.loaders import loader
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
# legends for multiple x-axis
import matplotlib.lines as mlines
import matplotlib
matplotlib.use('QtCairo')

from incense import ExperimentLoader

from IPython.core.debugger import set_trace

Array = np.ndarray
# FIGURE_X = 6.0
# FIGURE_Y = 4.0
# MEAN_CURVE_COLOR = (0.89, 0.282, 0.192)
# SMOOTHING_CURVE_COLOR = (0.33, 0.33, 0.33)
# SEED_PATTERN = r"seed=(.*?)\)"
# M_PATTERN = r"M=(.*?)\,"
FIGS_SAVE_DIR = "results/plots/t_test"


TAG_HYPERGROUP_NTWQL_QUERIES = OrderedDict({
    0: {
        'query_ids': [261, 262, 274],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_0',
            'config.networked_edges': 1,
            'config.networked_rounds': 1,
            'config.networked_interval': 1,
            'config.t_max': 2005000
        }
    },
    1: {
        'query_ids': [259, 271, 275],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_1',
            'config.networked_edges': 1,
            'config.networked_rounds': 1,
            'config.networked_interval': 5,
            'config.t_max': 2005000
        }
    },
    2: {
        'query_ids': [255, 260, 270],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_2',
            'config.networked_edges': 1,
            'config.networked_rounds': 1,
            'config.networked_interval': 10,
            'config.t_max': 2005000
        }
    },
    3: {
        'query_ids': [256, 269, 276],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_3',
            'config.networked_edges': 1,
            'config.networked_rounds': 5,
            'config.networked_interval': 1,
            'config.t_max': 2005000
        }
    },
    4: {
        'query_ids': [254, 257, 273],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_4',
            'config.networked_edges': 1,
            'config.networked_rounds': 5,
            'config.networked_interval': 5,
            'config.t_max': 2005000
        }
    },
    5: {
        'query_ids': [258, 268, 272],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_5',
            'config.networked_edges': 1,
            'config.networked_rounds': 5,
            'config.networked_interval': 10,
            'config.t_max': 2005000
        }
    },
    6: {
        'query_ids': [277, 278, 291],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_6',
            'config.networked_edges': 1,
            'config.networked_rounds': 10,
            'config.networked_interval': 1,
            'config.t_max': 2005000
        }
    },
    7: {
        'query_ids': [279, 286, 290],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_7',
            'config.networked_edges': 1,
            'config.networked_rounds': 10,
            'config.networked_interval': 5,
            'config.t_max': 2005000
        }
    },
    8: {
        'query_ids': [280, 281, 287],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_8',
            'config.networked_edges': 1,
            'config.networked_rounds': 10,
            'config.networked_interval': 10,
            'config.t_max': 2005000
        }
    },
    9: {
        'query_ids': [282, 288, 293],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_9',
            'config.networked_edges': 2,
            'config.networked_rounds': 1,
            'config.networked_interval': 1,
            'config.t_max': 2005000
        }
    },
    10: {
        'query_ids': [283, 284, 289],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_10',
            'config.networked_edges': 2,
            'config.networked_rounds': 1,
            'config.networked_interval': 5,
            'config.t_max': 2005000
        }
    },
    11: {
        'query_ids': [285, 292, 294],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_11',
            'config.networked_edges': 2,
            'config.networked_rounds': 1,
            'config.networked_interval': 10,
            'config.t_max': 2005000
        }
    },
    12: {
        'query_ids': [295, 296, 305],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_12',
            'config.networked_edges': 2,
            'config.networked_rounds': 5,
            'config.networked_interval': 1,
            'config.t_max': 2005000
        }
    },
    13: {
        'query_ids': [297, 304, 308],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_13',
            'config.networked_edges': 2,
            'config.networked_rounds': 5,
            'config.networked_interval': 5,
            'config.t_max': 2005000
        }
    },
    14: {
        'query_ids': [298, 299, 306],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_14',
            'config.networked_edges': 2,
            'config.networked_rounds': 5,
            'config.networked_interval': 10,
            'config.t_max': 2005000
        }
    },
    15: {
        'query_ids': [300, 307, 312],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_15',
            'config.networked_edges': 2,
            'config.networked_rounds': 10,
            'config.networked_interval': 1,
            'config.t_max': 2005000
        }
    },
    16: {
        'query_ids': [301, 302, 309],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_16',
            'config.networked_edges': 2,
            'config.networked_rounds': 10,
            'config.networked_interval': 5,
            'config.t_max': 2005000
        }
    },
    17: {
        'query_ids': [303, 310, 311],
        'source': 'remote',
        'query_config': {
            'config.name': 'ntwql',
            'config.hypergroup': 'hp_grp_17',
            'config.networked_edges': 2,
            'config.networked_rounds': 10,
            'config.networked_interval': 10,
            'config.t_max': 2005000
        }
    },
    # TODO:
    # 18: {
    #     'query_ids': [],
        # 'source': 'remote',
    #     'query_config': {
    #         'config.name': 'ntwql',
    #         'config.hypergroup': 'hp_grp_0',
    #         'config.networked_edges': 3,
    #         'config.networked_rounds': 1,
    #         'config.networked_interval': 1,
    #     }
    # },
    # 19: {
    #     'query_ids': [],
        # 'source': 'remote',
    #     'query_config': {
    #         'config.name': 'ntwql',
    #         'config.hypergroup': 'hp_grp_1',
    #         'config.networked_edges': 3,
    #         'config.networked_rounds': 1,
    #         'config.networked_interval': 5,
    #     }
    # },
    # 20: {
    #     'query_ids': [],
        # 'source': 'remote',
    #     'query_config': {
    #         'config.name': 'ntwql',
    #         'config.hypergroup': 'hp_grp_2',
    #         'config.networked_edges': 3,
    #         'config.networked_rounds': 1,
    #         'config.networked_interval': 10,
    #     }
    # },
})

def confidence_interval(samples: np.ndarray, num_resamples: int=20_000):
    resampled = np.random.choice(samples,
                                size=(len(samples), num_resamples),
                                replace=True)
    point_estimations = np.mean(resampled, axis=0)
    confidence_interval = [np.percentile(point_estimations, 2.5),
                           np.percentile(point_estimations, 97.5)]
    return confidence_interval

def main(environment: str, algoname: str, hypergroup: Dict):

    mongo_uri = mdb.build_conn('remote')
    loader = ExperimentLoader(mongo_uri=mongo_uri, db_name=mdb.MONGO_DB_NAME)
    path = Path(FIGS_SAVE_DIR)
    path.mkdir(exist_ok=True)
    def gt(x): return x['test_return_mean'][:, x['neighborhood']].flatten()
    data = defaultdict(list)
    samples = []
    for i, query in hypergroup.items():


        experiments = loader.find({
            '$and': [
                {'_id': { "$in": query['query_ids']}},
                query["query_config"],
            ]
        })

        values = []
        steps = []
        sample_size = 0
        for experiment in experiments:
            ts = experiment.metrics["test_return_mean"]
            values.append(ts.values)
            steps.append(ts.index)

            sample_size += 1

        assert sample_size==3, f'{i}-th hypergroup'
        test_return_mean = np.vstack(values)
        amax = np.argmax(test_return_mean.sum(axis=0))
        aub = min(amax + 3, test_return_mean.shape[1]) # upper limit is exclusive
        alb = max(amax - 2, 0)
        aub, alb = aub + (2 - (amax - alb)), alb - (3 - (aub - amax))

        samples.append({
            'task': environment,
            'label': f'TestHyperparameterGroup{i:02d}',
            'name': query['query_config']['config.name'],
            'experiment_id': query['query_ids'],
            'test_return_mean': test_return_mean,
            'neighborhood': slice(alb, aub)
        })

    path = path / environment
    path.mkdir(exist_ok=True)
    set_trace()
    for sample_ab in combinations(samples, 2):

        # The sample with the greatest mean is assigned to the sample a
        sample_a, sample_b = sorted(sample_ab, key=lambda x: gt(x).mean(), reverse=True)
        sample_return_a, sample_return_b = gt(sample_a), gt(sample_b)
        mean_a, mean_b = sample_return_a.mean(), sample_return_b.mean()
        exp_id_a, exp_id_b = sample_a['label'], sample_b['label']
        
        h0 = f'{mean_a:0.3f} <= {mean_b:0.3f}'
        h1 = f'{mean_a:0.3f} > {mean_b:0.3f}'
        # Defines the alternative hypothesis.
        #'less’: the mean of the distribution underlying the first sample is less than the mean of the distribution
        # equal_var: If False, perform Welch’s t-test, which does not assume equal population variance
        tt = stats.ttest_ind(
            a=sample_return_a,
            b=sample_return_b,
            alternative='greater',
            equal_var=False
        )

        data['task'].append(sample_a['task'])
        data['sample_a'].append(f"{exp_id_a}/{sample_a['name']}")
        data['sample_b'].append(f"{exp_id_b}/{sample_b['name']}")
        data['mean_a'].append(mean_a)
        data['mean_b'].append(mean_b)
        data['H0'].append(h0)
        data['H1'].append(h1)
        data['t-statistic'].append(tt.statistic)
        data['p-value'].append(tt.pvalue)

    df = pd.DataFrame.from_dict(data)
    fname = f"{algoname}_{environment.split(':')[-1]}.csv"
    print(df.head(), fname)
    df.to_csv(Path(FIGS_SAVE_DIR) / environment / fname)

if __name__ == "__main__":
    ENV = 'mpe:SimpleTag-v0'
    # ENV = 'rware:rware-tiny-4ag-v1'
    # ENV = 'lbforaging:Foraging-15x15-3p-5f-v1'
    main(ENV, 'ntwql', TAG_HYPERGROUP_NTWQL_QUERIES)
