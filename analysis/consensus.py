"""Plots sampled weights and biases per consensus step

* controls if weights tend to an average.
* shows that consensus happens on a shorter timescale than critic convergence
"""
import argparse
import json
from pathlib import Path
import re
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

FIGURE_X = 6.0
FIGURE_Y = 4.0
CONSENSUS_PATTERN = r"weight_(\d+)_\d+_\d+"
PLAYERS_PATTERN = r"weight_\d+_(\d+)_\d+"

parser = argparse.ArgumentParser()
parser.add_argument("metrics_json_path", type=str, help="String containing the path")
parser.add_argument("--step", "-s", type=int, default=None, help="The training step. (default: random)")
parser.add_argument("--weight", "-w", type=int, choices=(0, 3, 7), default=None, help="The training weight. (default: random)")

def get_opt() -> Tuple[Path, Dict, Dict]:
    """Sanitize input parameters

    Returns:
    -------
    weights: Dict
    biases: Dict
    info: Dict
    """
    args = parser.parse_args()
    path = Path(args.metrics_json_path)
    step = args.step
    weight = args.weight
    task = path.parent.parent.stem.split(':')[-1]

    # test if file exists
    if not path.is_file():
        raise ValueError(f"{path.as_posix()} Doesn't point to a valid metric.json file.")
    with path.open('rt') as f:
        metrics = json.load(f)
    total_steps = len(metrics['return_mean']['steps'])
    if step is None:
        step = np.random.randint(total_steps)
    elif step >= total_steps:
        raise ValueError(f'Step < total_steps got {step} and {total_steps}.')

    if weight is None:
        weight = np.random.choice((0, 3, 7))
    elif weight not in (0, 3, 7):
        raise ValueError('Weight must be in (0, 3, 7)')
    pattern = re.compile(f'.*weight_\d+_\d+_{weight}')



    def fw(x):      # filter weight
        return pattern.match(x[0]) is not None
    def mwb(x):      # map weight or bias step
        return (x[0], {kx: vx[step] for kx, vx in x[1].items()})
    weights = dict(map(mwb, filter(fw, metrics.items())))

    def fb(x):      # filter bias
        return 'bias' in x[0]
    biases = dict(map(mwb, filter(fb, metrics.items())))

    def fv(x):      # filter values
        return 'v_final_mean_' in x[0]
    values = dict(map(mwb, filter(fv, metrics.items())))

    info = {'weight': weight, 'task': task, 'step': step}
    return weights, biases, values, info

# TODO: place data within info dict
def plots(weights: Dict, biases: Dict, values: Dict, info: Dict) -> None:
    """Plots consensus rounds """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
    cr = info['consensus_rounds'] + 1
    pl = info['num_players'] + 1
    wid = info['weight']

    X = np.arange(cr)
    Y = np.zeros((cr, pl))

    title = f"{info['task']}"
    labels = [f'player {y}' for y in range(pl)]
    # for z in ('weight', 'bias', 'v_final_mean_player'):
    for z in ('v_final_mean_player',):
        if z == 'weight':
            data = weights
        elif  z == 'bias':
            data = biases
        elif z == 'v_final_mean_player':
            data = values
        else:
            raise ValueError(f'Unknown data type')

        for x in range(cr):
            for y in range(pl):
                key = f'{z}_{x}_{y}'
                if z == 'weight':
                    key += f'_{wid}'
                elif z == 'bias':
                    key +='_0'
                Y[x, y] = data[key]['values']
        timestep = data[key]['steps']     # its the same for all samples.
         
        
        ylabel = z.title()
        if z == 'weight':
            ylabel += f'[{wid}]'
        print(Y, Y.mean(axis=1))
        plt.plot(X, Y, label=labels)
        plt.plot(X, Y.mean(axis=1), linestyle='dashed', label='target')
        plt.xlabel(f"Consensus Rounds (step: {timestep})")
        plt.ylabel(ylabel)
        plt.legend(loc=4)
        plt.suptitle(title)
        plt.show()


def main():
    weights, biases, values, info = get_opt()

    # determines the number of consensus rounds
    # pattern = re.compile('weight_(\d+)_\d+_\d+')
    def mtchr(x):      # map consensus rounds
        return int(re.search(CONSENSUS_PATTERN, x).group(1))
    # info['consensus_rounds'] = max(map(mtchr, list(weights.keys())))

    info['consensus_rounds'] = 5

    # determines the number of players
    # pattern = re.compile('weight_\d+_(\d+)_\d+')
    def mtchr(x):      # map number of players
        return int(re.search(PLAYERS_PATTERN, x).group(1))
    # info['num_players'] = max(map(mtchr, list(weights.keys())))
    info['num_players'] = 3 - 1
    plots(weights, biases, values, info)

if __name__ == '__main__':
    main()
