"""Non-parametric confidence interval with respect to test_return_mean"""
from collections import defaultdict
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path
import json

import matplotlib
from functools import partial
from utils import standard_error



##################################################################
# Script parameters:
FIGS_SAVE_DIR = "results/plots/nonparam_ci"

EXP_IDS = {

    # "lbforaging:Foraging-2s-10x10-3p-3f-v1": {
    #     "ia2c_ns": {
    #         "nonlinear_critic-validation": list(range(3098, 3102 + 1)),
    #     },
    #     "inda2c": {
    #         "nonlinear_critic-validation": list(range(3098, 3102 + 1)),
    #     },
    #     "ntwa2c": {
    #         "nonlinear_critic-validation": list(range(3073, 3077 + 1)),
    #     },
    # },
    "lbforaging:Foraging-15x15-3p-5f-v1": {
        "inda2c": {
            "nonlinear_critic-debug": list(range(3098, 3102 + 1)),
        },
        "ntwa2c": {
            "nonlinear_critic-debug": list(range(3073, 3077 + 1)),
        },
    },
}

COLORS = {
    "ia2c_ns": '#377eb8',
    "inda2c": '#ff7f00',
    "ntwa2c": '#4daf4a',
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
            test_ids = []
            for exp_id, rng in values.items():
                mpath = mpath / exp_id / scenario
                for path in sorted(
                    mpath.rglob('*/metrics.json'), key=lambda x: int(x.parent.stem)
                ):
                    test_ids.append(int(path.parent.stem))
                    with path.open("r") as f:
                        data = json.load(f)
                    experiments.append(data['test_return_mean']['values'])
            ret.append({
                'task': scenario,
                'label': model,
                'test_return_mean': np.vstack(experiments),
                'test_ids': test_ids
            })
    return ret
        
# Print script arguments.
print("Args.:")

print("Save dir:", FIGS_SAVE_DIR)

ALGO_ID_TO_ALGO_LBL = {"ia2c_ns": "CENTRAL", "inda2c": "IL", "ntwa2c": "NTWRKD"}

RUNS = []

if RENDER_LATEX:
    matplotlib.rcParams['text.usetex'] = True
    plt.rc('text.latex', preamble=r'\usepackage{pifont}')
    matplotlib.rcParams.update({'font.size': 18})

def confidence_interval(samples: np.ndarray, num_resamples: int=20_000):
    # resampled = np.random.choice(samples,
    #                             size=(len(samples), num_resamples),
    #                             replace=True)
    # point_estimations = np.mean(resampled, axis=0)
    # confidence_interval = [np.percentile(point_estimations, 2.5),
    #                        np.percentile(point_estimations, 97.5)]
    ci = standard_error(samples.std(), len(samples.flatten()), 0.95)

    return [ci * 0.5] * 2


def main():

    path = Path()
    if FIGS_SAVE_DIR:
        path = Path(FIGS_SAVE_DIR)
        path.mkdir(exist_ok=True)

    # Load data.
    RUNS = loader()
    ################################################################
    # Get scalar performance values.
    ################################################################
    d = defaultdict(list)
    def tsk(x): return x['task']
    def eq(y, x): return tsk(x) == y
    tasks = sorted(set(map(tsk, RUNS)))
    for task in tasks:
        path = path / task
        path.mkdir(exist_ok=True)
        task_data = filter(partial(eq, task), RUNS)
        labels = []
        for data in task_data:
            algo_data = data["test_return_mean"]
            scalar = np.mean(algo_data[:, -5:])
            return_mean_data = np.mean(algo_data[:, -5:], axis=1)

            ci = confidence_interval(return_mean_data)
            d['algorithm'].append(data['label'])
            d['u_mean'].append(scalar)
            d['u_upper_ci'].append(ci[1])
            d['u_lower_ci'].append(ci[0])

            labels.append(data['label'])
        # Dump to csv file.
        df = pd.DataFrame.from_dict(d)
        fname = f"{'_'.join(labels)}.csv"
        print(df.head(), fname)
        df.to_csv(Path(FIGS_SAVE_DIR) / task / fname)
        # df_name = f"{ENV}_{ALGO_ID}"
        # df.to_csv(f'{FIGS_SAVE_DIR}/{df_name}.csv')

        ################################################################
        # Plot unif. performance (bar chart).
        ################################################################
        fig = plt.figure()
        fig.set_size_inches(7.0, 5.0)
        fig.tight_layout()

        X = np.arange(len(RUNS))

        errors = []
        ys = []
        for algo_data in RUNS:
            dict_key = "test_return_mean"
            y = np.mean(algo_data[dict_key][:, -5:])
            ys.append(y)
            return_mean_data = np.mean(algo_data[dict_key][:, -5:], axis=1)
            ci = confidence_interval(return_mean_data)

            # error_top = ci[1] - y
            # error_bottom = y - ci[0]
            error_top = y + ci[1]
            error_bottom = y - ci[0]
            errors.append([error_top, error_bottom])

        errors = np.array(errors).T
        plt.errorbar(X, ys,
            yerr=errors,
            fmt='o',
            capsize=6,
            zorder=50,
        )
        plt.xticks(X, labels=labels)

        plt.grid()
        plt.ylabel("Return")

        if FIGS_SAVE_DIR:
            fname = f"barchart_{'_'.join(labels)}.pdf"
            plt.savefig(Path(FIGS_SAVE_DIR) / task / fname, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()

if __name__ == "__main__":
    main()
