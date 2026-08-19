"""Get model parameters.

This script extracts the model parameter for the regression models.

    1. Load data
    2. Run model
    3. Save data
"""


import os
import platform

import matplotlib

# Simple cross-platform backend selection
if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")  # Headless
elif platform.system() == "Darwin":
    matplotlib.use("MacOSX")  # macOS native
else:
    matplotlib.use("Qt5Agg")  # Linux with display, Windows, others

import sys

import numpy as np
import pandas as pd

from FOR_1_Paper.for_utilities import safe_save_dataframe
from FOR_1_Paper.modeling.simulation_rbm import simulation_loop

# Set random number generator for reproducible results
np.random.seed(123)

# ------------
# 1. Load data
# ------------

# Load preprocessed data
df_exp = pd.read_pickle("for_data/data_prepr.pkl")
n_subj = len(np.unique(df_exp["subj_num"]))  # number of subjects

# ------------
# 2. Run model
# ------------

# Simulation parameters
model = pd.DataFrame(
    columns=["omikron_0", "omikron_1", "h", "s", "u", "sigma_H", "subj_num"]
)
model.loc[:, "omikron_0"] = np.repeat(1, n_subj)
model.loc[:, "omikron_1"] = np.repeat(0, n_subj)
model.loc[:, "lambda_0"] = np.nan
model.loc[:, "lambda_1"] = np.nan
model.loc[:, "h"] = np.repeat(0.1, n_subj)
model.loc[:, "s"] = np.repeat(1, n_subj)
model.loc[:, "u"] = np.repeat(0, n_subj)
model.loc[:, "sigma_H"] = np.repeat(0.0001, n_subj)
model.loc[:, "subj_num"] = np.arange(n_subj) + 1

n_sim = 1  # 1 simulation per subject
sim_pers = False  # no perseveration
all_est_errs, all_data = simulation_loop(
    df_exp, model, n_subj, plot_data=False, n_sim=n_sim, sim=False
)

# Test if subject numbers still line up
comp_subj_num = df_exp["subj_num"] == all_data["subj_num"]
if False in comp_subj_num.values:
    sys.exit("Sub IDs don't match!")

# ------------
# 3. Save data
# ------------

df_exp = (
    pd.concat([df_exp, all_data.drop(["subj_num", "ID", "mu_t_rad"], axis=1)], axis=1)
    .drop_duplicates()
    .reset_index(drop=True)
)
df_exp.name = "data_prepr_model"

# Save data
safe_save_dataframe(df_exp)
