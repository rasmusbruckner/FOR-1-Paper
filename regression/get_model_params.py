"""Get model parameters: Extracts the model parameter for the regression models for the FOR and Vaghi datasets."""

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


# Function to extract model parameters
def get_model_params(
    df_dataset: pd.DataFrame, n_subj_dataset: int, hazard_rate: float = 0.1
):
    """Extracts the model parameters.

    Parameters
    ----------
    df_dataset : pd.DataFrame
        Current dataset (FOR or Vaghi).
    n_subj_dataset : int
        Number of subjects in the dataset.
    hazard_rate : float
        Hazard rate of the experiment (default: 0.1 for the FOR dataset)

    Returns
    -------
    pd.DataFrame

    """

    # Set a random number generator for reproducible results
    np.random.seed(123)

    # ---------
    # Run model
    # ---------

    # Simulation parameters
    model = pd.DataFrame(
        columns=[
            "omikron_0",
            "omikron_1",
            "lambda_0",
            "lambda_1",
            "h",
            "s",
            "u",
            "sigma_H",
            "subj_num",
        ]
    )
    model.loc[:, "omikron_0"] = np.repeat(1, n_subj_dataset)
    model.loc[:, "omikron_1"] = np.repeat(0, n_subj_dataset)
    model.loc[:, "lambda_0"] = np.nan
    model.loc[:, "lambda_1"] = np.nan
    model.loc[:, "h"] = np.repeat(hazard_rate, n_subj_dataset)
    model.loc[:, "s"] = np.repeat(1, n_subj_dataset)
    model.loc[:, "u"] = np.repeat(0, n_subj_dataset)
    model.loc[:, "sigma_H"] = np.repeat(0.0001, n_subj_dataset)
    model.loc[:, "subj_num"] = np.arange(n_subj_dataset) + 1

    n_sim = 1  # 1 simulation per subject
    all_est_errs, all_data = simulation_loop(
        df_dataset, model, n_subj_dataset, plot_data=False, n_sim=n_sim, sim=False
    )

    # Test if subject numbers still line up
    comp_subj_num = df_dataset["subj_num"] == all_data["subj_num"]
    if False in comp_subj_num.values:
        sys.exit("Sub IDs don't match!")

    # ----------
    # Merge data
    # ----------

    df_dataset = (
        pd.concat(
            [df_dataset, all_data.drop(["subj_num", "ID", "mu_t_rad"], axis=1)], axis=1
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df_dataset


# --------------
# 1. FOR dataset
# --------------

# Load preprocessed data
df_exp = pd.read_pickle("for_data/data_prepr.pkl")
n_subj = len(np.unique(df_exp["subj_num"]))

# Extract model parameters
df_exp = get_model_params(df_exp, n_subj, hazard_rate=0.1)

# Save FOR data
df_exp.name = "data_prepr_model"
safe_save_dataframe(df_exp)

# ----------------
# 2. Vaghi dataset
# ----------------

df_vaghi = pd.read_pickle("for_data/vaghi_data_prepr.pkl")
n_subj = len(np.unique(df_vaghi["subj_num"]))

# Extract model parameters
df_vaghi = get_model_params(df_vaghi, n_subj, hazard_rate=0.125)

# Save Vaghi data
df_vaghi.name = "vaghi_data_prepr_model"
safe_save_dataframe(df_vaghi)
