"""Two Sample t-test in Python"""
import numpy as np

from collections import defaultdict
import pandas as pd
import scipy.stats as stats
from pathlib import Path
import json
from itertools import combinations
from functools import partial


ALGO_ID_TO_ALGO_LBL = {"ia2c_ns": "CENTRAL", "inda2c": "IL", "ntwa2c": "NTWRKD"}

##################################################################
# Script parameters:
ENV = "LBF"
ALGO_ID = "ntwa2c"
EPISODE_LENGTH = 50 # Needs to be the same for all algorithms.
FIGS_SAVE_DIR = "results/plots/t_test"

# THIS SEEMS TO BE A REFERENCE.
EXP_IDS = {
    "lbforaging:Foraging-15x15-3p-5f-v1": {
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
                mpath = mpath / exp_id / scenario
                for path in sorted(
                    mpath.rglob('*/metrics.json'), key=lambda x: int(x.parent.stem)
                ):
                    test_ids.append(int(path.parent.stem))
                    with path.open("r") as f:
                        data = json.load(f)
                    experiments.append(data['test_return_mean']['values'])

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

            # The sample with the greatest mean is assigned to the sample a
            sample_a, sample_b = sorted(sample_ab, key=lambda x: gt(x).mean(), reverse=True)
            sample_return_a, sample_return_b = gt(sample_a), gt(sample_b)
            mean_a, mean_b = sample_return_a.mean(), sample_return_b.mean()
            
            h0 = f'{mean_a:0.3f} = {mean_b:0.3f}'
            h1 = f'{mean_a:0.3f} > {mean_b:0.3f}'
            tt = stats.ttest_ind(a=sample_return_a, b=sample_return_b)

            d['task'].append(sample_a['task'])
            d['sample_a'].append(sample_a['label'])
            d['sample_b'].append(sample_b['label'])
            d['mean_a'].append(mean_a)
            d['mean_b'].append(mean_b)
            d['H0'].append(h0)
            d['H1'].append(h1)
            d['t-statistic'].append(tt.statistic)
            d['t-pvalue'].append(tt.pvalue)

        labels = sorted(set(map(lambda x: x['label'], RUNS)))
        df = pd.DataFrame.from_dict(d)
        fname = f"{'_'.join(labels)}.csv"
        print(df.head(), fname)
        df.to_csv(Path(FIGS_SAVE_DIR) / task / fname)



if __name__ == "__main__":
    main()
