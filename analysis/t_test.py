"""Two Sample t-test in Python"""
<<<<<<< HEAD
import os
=======
>>>>>>> search
import numpy as np

from collections import defaultdict
import pandas as pd
import scipy.stats as stats
<<<<<<< HEAD
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style("dark")
from pathlib import Path
import json
from itertools import combinations

import matplotlib
=======
from pathlib import Path
import json
from itertools import combinations
from functools import partial

>>>>>>> search

ALGO_ID_TO_ALGO_LBL = {"ia2c_ns": "CENTRAL", "inda2c": "IL", "ntwa2c": "NTWRKD"}

##################################################################
# Script parameters:
ENV = "LBF"
ALGO_ID = "ntwa2c"
EPISODE_LENGTH = 50 # Needs to be the same for all algorithms.
FIGS_SAVE_DIR = "results/plots/t_test"
<<<<<<< HEAD
EXPERIMENT_ID = 'consensus'
=======
>>>>>>> search

# THIS SEEMS TO BE A REFERENCE.
EXP_IDS = {
    "lbforaging:Foraging-15x15-3p-5f-v1": {
<<<<<<< HEAD
        "ntwa2c": {
            "hp_grp_00": list(range(3098, 3102 + 1)),
            "hp_grp_01": list(range(3098, 3102 + 1)),
            "hp_grp_02": list(range(3098, 3102 + 1)),
            "hp_grp_03": list(range(3098, 3102 + 1)),
            "hp_grp_04": list(range(3098, 3102 + 1)),
            "hp_grp_05": list(range(3098, 3102 + 1)),
            "hp_grp_06": list(range(3098, 3102 + 1)),
            "hp_grp_07": list(range(3098, 3102 + 1)),
            "hp_grp_08": list(range(3098, 3102 + 1)),
            "hp_grp_09": list(range(3098, 3102 + 1)),
            "hp_grp_10": list(range(3098, 3102 + 1)),
            "hp_grp_11": list(range(3098, 3102 + 1)),
            "hp_grp_12": list(range(3098, 3102 + 1)),
            "hp_grp_13": list(range(3098, 3102 + 1)),
            "hp_grp_14": list(range(3098, 3102 + 1)),
            "hp_grp_15": list(range(3098, 3102 + 1)),
            "hp_grp_16": list(range(3098, 3102 + 1)),
            "hp_grp_18": list(range(3098, 3102 + 1)),
            "hp_grp_19": list(range(3098, 3102 + 1)),
            "hp_grp_20": list(range(3098, 3102 + 1)),
        },
        # "ia2c_ns": {
        #     "30_000_000": list(range(3098, 3102 + 1)),
        # },
        # "ntwa2c": {
        #     "nonlinear_critic-validation": list(range(3073, 3077 + 1)),
        # },
    },
}
# EXP_IDS = {
#     "lbforaging:Foraging-15x15-3p-5f-v1": {
#         "ia2c_ns": {
#             "25_000_000": list(range(3098, 3102 + 1)),
#             "30_000_000": list(range(3098, 3102 + 1)),
#         },
#         # "ia2c_ns": {
#         #     "30_000_000": list(range(3098, 3102 + 1)),
#         # },
#         # "ntwa2c": {
#         #     "nonlinear_critic-validation": list(range(3073, 3077 + 1)),
#         # },
#     },
# }
=======
        "ia2c_ns": {
            "nonlinear_critic-validation": list(range(3098, 3102 + 1)),
        },
        "inda2c": {
            "nonlinear_critic-validation": list(range(3098, 3102 + 1)),
        },
        "ntwa2c": {
            "nonlinear_critic-validation": list(range(3073, 3077 + 1)),
        },
    },
}
>>>>>>> search

RENDER_LATEX = False

# LOADS EXPERIMENTS
BASE_PATH = Path(f"results/sacred/")
def loader():
    ret = []
    for scenario, values in EXP_IDS.items():
        for model, values in values.items():
            mpath = BASE_PATH / model
            experiments = []
            test_ids  = []
            for exp_id, rng in values.items():
<<<<<<< HEAD
                epath = mpath / exp_id / scenario
                for path in sorted(
                    epath.rglob('*/metrics.json'), key=lambda x: int(x.parent.stem)
=======
                mpath = mpath / exp_id / scenario
                for path in sorted(
                    mpath.rglob('*/metrics.json'), key=lambda x: int(x.parent.stem)
>>>>>>> search
                ):
                    test_ids.append(int(path.parent.stem))
                    with path.open("r") as f:
                        data = json.load(f)
                    experiments.append(data['test_return_mean']['values'])
<<<<<<< HEAD
                    print(f'{scenario}/{model}/{exp_id}/{test_ids[-1]}')


                test_return_mean = np.vstack(experiments)
                amax = np.argmax(test_return_mean.sum(axis=0))
                aub = min(amax + 3, test_return_mean.shape[1]) # upper limit is exclusive
                alb = max(amax - 2, 0)
                aub, alb = aub + (2 - (amax - alb)), alb - (3 - (aub - amax))
                ret.append({
                    'task': scenario,
                    'label': model,
                    'experiment_id': exp_id,
                    'test_return_mean': test_return_mean,
                    'test_ids': test_ids,
                    'neighborhood': slice(alb, aub)
                })

                experiments = []
                test_ids = []
=======

            test_return_mean = np.vstack(experiments)
            amax = np.argmax(test_return_mean.sum(axis=0))
            aub = min(amax + 3, test_return_mean.shape[1]) # upper limit is exclusive
            alb = max(amax - 2, 0)
            aub, alb = aub + (2 - (amax - alb)), alb - (3 - (aub - amax))
            ret.append({
                'task': scenario,
                'label': model,
                'test_return_mean': test_return_mean,
                'test_ids': test_ids,
                'neighborhood': slice(alb, aub)
            })
>>>>>>> search
    return ret
        
##################################################################

# Print script arguments.
print("Args.:")
print("Environment:", ENV)
print("Algorithm:", ALGO_ID)
# print("Runs to plot:", RUNS_TO_PLOT)
print("Save dir:", FIGS_SAVE_DIR)


COLORS = {
    "Obs.": '#377eb8',
    # "J. obs.": '#ff7f00',
    # "MD": '#4daf4a',
    # "MARO": '#f781bf',
    # "Pred.\ding{51}, Train\ding{55}": '#a65628', # Pred TRUE, Train FALSE
    # "Pred.\ding{55}, Train\ding{51}": '#984ea3', # Pred FALSE, Train TRUE
    # "Pred.\ding{55}, Train\ding{55}": '#999999', # Pred FALSE, Train FALSE
    # "p = 0,1": '#e41a1c',
    # "p = 0.5": '#dede00',
    # "p = 1": '#a65628',
}

if RENDER_LATEX:
    matplotlib.rcParams['text.usetex'] = True
    plt.rc('text.latex', preamble=r'\usepackage{pifont}')
    matplotlib.rcParams.update({'font.size': 18})

def confidence_interval(samples: np.ndarray, num_resamples: int=20_000):
    resampled = np.random.choice(samples,
                                size=(len(samples), num_resamples),
                                replace=True)
    point_estimations = np.mean(resampled, axis=0)
    confidence_interval = [np.percentile(point_estimations, 2.5),
                           np.percentile(point_estimations, 97.5)]
    return confidence_interval

def main():
<<<<<<< HEAD
    samples = loader()
    path = Path(FIGS_SAVE_DIR)
    path.mkdir(exist_ok=True)
    def gt(x): return x['test_return_mean'][:, x['neighborhood']].flatten()
    d = defaultdict(list)
    tasks = sorted(set(map(lambda x: x['task'], samples)))
    for task in tasks:
        path = path / task
        path.mkdir(exist_ok=True)
        for sample_ab in combinations(samples, 2):
=======
    RUNS = loader()
    path = Path(FIGS_SAVE_DIR)
    path.mkdir(exist_ok=True)
    def gt(x): return x['test_return_mean'][:, x['neighborhood']].flatten()
    def tsk(x): return x['task']
    def eq(y, x): return tsk(x) == y
    d = defaultdict(list)
    tasks = sorted(set(map(tsk, RUNS)))
    for task in tasks:
        path = path / task
        path.mkdir(exist_ok=True)
        task_runs = filter(partial(eq, task), RUNS)
        for sample_ab in combinations(task_runs, 2):
>>>>>>> search

            # The sample with the greatest mean is assigned to the sample a
            sample_a, sample_b = sorted(sample_ab, key=lambda x: gt(x).mean(), reverse=True)
            sample_return_a, sample_return_b = gt(sample_a), gt(sample_b)
            mean_a, mean_b = sample_return_a.mean(), sample_return_b.mean()
<<<<<<< HEAD
            exp_id_a, exp_id_b = sample_a['experiment_id'], sample_b['experiment_id']
            
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

            d['task'].append(sample_a['task'])
            d['sample_a'].append(f"{exp_id_a}/{sample_a['label']}")
            d['sample_b'].append(f"{exp_id_b}/{sample_b['label']}")
=======
            
            h0 = f'{mean_a:0.3f} = {mean_b:0.3f}'
            h1 = f'{mean_a:0.3f} > {mean_b:0.3f}'
            tt = stats.ttest_ind(a=sample_return_a, b=sample_return_b)

            d['task'].append(sample_a['task'])
            d['sample_a'].append(sample_a['label'])
            d['sample_b'].append(sample_b['label'])
>>>>>>> search
            d['mean_a'].append(mean_a)
            d['mean_b'].append(mean_b)
            d['H0'].append(h0)
            d['H1'].append(h1)
            d['t-statistic'].append(tt.statistic)
<<<<<<< HEAD
            d['p-value'].append(tt.pvalue)

        # labels = sorted(set(map(lambda x: f"{x['experiment_id']}-{x['label']}", RUNS)))
        df = pd.DataFrame.from_dict(d)
        # fname = f"{'_'.join(labels)}.csv"
        fname = f"{EXPERIMENT_ID}.csv"
=======
            d['t-pvalue'].append(tt.pvalue)

        labels = sorted(set(map(lambda x: x['label'], RUNS)))
        df = pd.DataFrame.from_dict(d)
        fname = f"{'_'.join(labels)}.csv"
>>>>>>> search
        print(df.head(), fname)
        df.to_csv(Path(FIGS_SAVE_DIR) / task / fname)



if __name__ == "__main__":
    main()
