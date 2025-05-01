"""Posterior predictive checks.

1. Load data
2. Run model
3. Save data
"""

import numpy as np
import pandas as pd

from FOR_1_Paper.for_modeling.for_simulation_rbm import simulation_loop
from FOR_1_Paper.for_utilities import safe_save_dataframe

# Set random number generator for reproducible results
np.random.seed(123)

# ------------
# 1. Load data
# ------------

# Load preprocessed data
df_exp = pd.read_pickle("for_data/data_prepr.pkl")
n_subj = len(np.unique(df_exp["subj_num"]))  # number of subjects

# Load estimated model parameters
df_estimates = pd.read_pickle("for_data/for_estimates_5_sp.pkl")

# Simulation parameters
df_model = pd.DataFrame(
    columns=["omikron_0", "omikron_1", "h", "s", "u", "sigma_H", "subj_num"]
)
df_model.loc[:, "omikron_0"] = df_estimates["omikron_0"].to_numpy()
df_model.loc[:, "omikron_1"] = df_estimates["omikron_1"].to_numpy()
df_model.loc[:, "h"] = df_estimates["h"].to_numpy()
df_model.loc[:, "s"] = df_estimates["s"].to_numpy()
df_model.loc[:, "u"] = df_estimates["u"].to_numpy()
df_model.loc[:, "sigma_H"] = df_estimates["sigma_H"].to_numpy()
df_model.loc[:, "subj_num"] = df_estimates["subj_num"].to_numpy()

# ------------
# 2. Run model
# ------------

n_sim = 1  # 1 simulation per subject
sim_pers = False  # no perseveration
all_est_errs, all_data = simulation_loop(
    df_exp, df_model, n_subj, plot_data=True, n_sim=n_sim, sim=True
)

# ------------
# 3. Save data
# ------------

all_est_errs.name = "post_pred_est_errs"
safe_save_dataframe(all_est_errs)
