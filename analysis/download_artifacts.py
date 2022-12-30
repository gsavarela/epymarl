
import os
from incense import ExperimentLoader
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
import tempfile
import shutil

import matplotlib

import src.utils.mongodb_utils as mongodb_utils

IDS = [2554, 2555, 2556, 2557, 2558]

ARTIFACTS_SAVE_DIR = "/home/ppsantos/git/hybrid-marl/results/rebuttal_exps_data/"


def main():

    print(ARTIFACTS_SAVE_DIR)

    # Load data.
    loader = ExperimentLoader(mongo_uri=mongodb_utils.get_db_uri(), db_name='vascocarvalhosantos')

    for id_number in IDS:

        # Path
        path = os.path.join(ARTIFACTS_SAVE_DIR, str(id_number), "50")
        os.makedirs(path, exist_ok=True)
        print(path)

        exp = loader.find_by_id(id_number)
        for artifact in exp.artifacts.values():
            print(artifact)
            artifact.save(path)

        os.rename(path + f"/{id_number}_agent.th", path + "/agent.th")
        os.rename(path + f"/{id_number}_opt.th", path + "/opt.th")
        os.rename(path + f"/{id_number}_mixer.th", path + "/mixer.th")
        os.rename(path + f"/{id_number}_perceptual_model.th", path + "/perceptual_network.th")
        os.rename(path + f"/{id_number}_perceptual_opt.th", path + "/perceptual_opt.th")

if __name__ == "__main__":
    main()