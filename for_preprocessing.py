"""Data Preprocessing: This script performs the preprocessing for the FOR paper."""

import sys

import numpy as np
import pandas as pd
from allinpy import get_file_paths, load_data, sorted_nicely

from for_utilities import safe_save_dataframe


def preprocessing():
    """This function loads and preprocesses the adaptive learning BIDS data for further analyses."""

    # Data folder
    folder_path = "for_data/for_bids_data"

    # Get all file names
    identifier = "*behav.tsv"
    file_paths = get_file_paths(folder_path, identifier)

    # Sort all file names according to participant ID
    file_paths = sorted_nicely(file_paths)

    # Load pseudonomized BIDS data
    data = load_data(file_paths)

    # -----------------------------
    # Update variables for analyses
    # -----------------------------

    # Extract block indexes
    new_block = data["new_block"].values  # block change indicator

    # Recode last entry in variables of interest of each block to nan
    # to avoid that data of different participants are mixed
    to_nan = np.zeros(len(new_block))
    to_nan[:-1] = new_block[1:]
    to_nan[-1] = 1  # manually added, because no new block after last trial

    # Prediction error
    data.loc[to_nan == 1, "delta_t"] = np.nan
    data["delta_t_rad"] = np.deg2rad(data["delta_t"])

    # Absolute estimation error
    data["e_t"] = abs(data["e_t"])
    data["e_t_rad"] = np.deg2rad(data["e_t"])

    # Update: On trial t, we analyze the difference between belief at t and t+1
    a_t = np.full(len(data), np.nan)
    a_t[:-1] = data["a_t"][1:].copy()
    a_t[to_nan == 1] = np.nan
    data["a_t"] = a_t
    data["a_t_rad"] = np.deg2rad(data["a_t"])

    # Prediction in radians
    data["b_t_rad"] = np.deg2rad(data["b_t"])

    # Mean in radians
    data["mu_t_rad"] = np.deg2rad(data["mu_t"])

    # Outcome in radians
    data["x_t_rad"] = np.deg2rad(data["x_t"])

    # Noise-level dummy variable
    data["kappa_dummy"] = np.nan
    data.loc[data["kappa_t"] == 8, "kappa_dummy"] = 1
    data.loc[data["kappa_t"] == 16, "kappa_dummy"] = -1

    # Concentration to variance
    vm_var = 1.0 / data["kappa_t"]
    data["sigma"] = np.sqrt(vm_var)

    # Catch dummy
    data["hit_dummy"] = np.nan
    data.loc[data["r_t"] == 1, "hit_dummy"] = 1
    data.loc[data["r_t"] == 0, "hit_dummy"] = -1

    # Perseveration: pers := 1, if a_t=0; 0, else
    data["pers"] = a_t == 0

    # Add information on group (here all in the same group)
    data["group"] = 0

    # Exclude subject 70 (20031)
    # Todo: provide justification in other script (basically showing weird behavior)
    data = data[~data["ID"].isin([20031])].reset_index(drop=True)

    # Add 1 to match Matlab integer logic (starting from 1 instead of 0)
    data["subj_num"] = pd.factorize(data["subj_num"])[0] + 1

    # Test if expected values appear in preprocessed data frames
    # ----------------------------------------------------------

    # Extrac number of subjects
    all_id = list(set(data["subj_num"]))  # ID for each participant
    n_subj = len(all_id)  # number of participants

    # Cycle over subjects
    for i in range(n_subj):

        # Extract data of current subject
        df_subj = data[(data["subj_num"] == i + 1)].copy()

        # Check expected values
        if not np.sum(np.isnan(df_subj["subj_num"])) == 0:
            sys.exit("Unexpected NaNs in subj_num")
        if not np.sum(np.isnan(df_subj["block"])) == 0:
            sys.exit("Unexpected NaNs in block")
        if not np.sum(df_subj["new_block"]) == 8:
            sys.exit("Unexpected NaNs in new_block")
        if not np.sum(np.isnan(df_subj["x_t"])) == 0:
            sys.exit("Unexpected NaNs in x_t")
        if not np.sum(np.isnan(df_subj["x_t_rad"])) == 0:
            sys.exit("Unexpected NaNs in x_t_rad")
        if not np.sum(np.isnan(df_subj["b_t"])) == 0:
            sys.exit("Unexpected NaNs in b_t")
        if not np.sum(np.isnan(df_subj["delta_t"])) == 8:
            sys.exit("Unexpected NaNs in delta_t")
        if not np.sum(np.isnan(df_subj["a_t"])) == 8:
            sys.exit("Unexpected NaNs in a_t")
        if not np.sum(np.isnan(df_subj["e_t"])) == 0:
            sys.exit("Unexpected NaNs in e_t")
        if not np.sum(np.isnan(df_subj["mu_t"])) == 0:
            sys.exit("Unexpected NaNs in mu_t")
        if not np.sum(np.isnan(df_subj["mu_t_rad"])) == 0:
            sys.exit("Unexpected NaNs in mu_t_rad")
        if not np.sum(np.isnan(df_subj["c_t"])) == 0:
            sys.exit("Unexpected NaNs in c_t")
        if not np.sum(np.isnan(df_subj["tac"])) == 0:
            sys.exit("Unexpected NaNs in tac")
        if not np.sum(np.isnan(df_subj["r_t"])) == 0:
            sys.exit("Unexpected NaNs in r_t")
        if not np.sum(np.isnan(df_subj["kappa_t"])) == 0:
            sys.exit("Unexpected NaNs in kappa_t")
        if not np.sum(np.isnan(df_subj["v_t"])) == 0:
            sys.exit("Unexpected NaNs in v_t")
        if not np.sum(np.isnan(df_subj["RT"])) == 0:
            sys.exit("Unexpected NaNs in RT")
        # Todo: Adjust initRT = NaN is some cases (when doing diffusion model)
        # initRT = RT was added after HH confetti pilot
        # for cases in which button is pressed very quickly
        # if not np.sum(np.isnan(df_subj["initRT"])) == 0:
        # sys.exit("Unexpected NaNs in initRT")
        # print(np.sum(np.isnan(df_subj["initRT"])))
        # if not np.sum(np.isnan(df_subj["initTend"])) == 0:
        # sys.exit("Unexpected NaNs in initRT")
        # print(np.sum(np.isnan(df_subj["initRT"])))
        # print(np.sum(np.isnan(df_subj["initTend"])))
        if not np.sum(np.isnan(df_subj["trial"])) == 0:
            sys.exit("Unexpected NaNs in trial")
        if not np.sum(np.isnan(df_subj["delta_t_rad"])) == 8:
            sys.exit("Unexpected NaNs in delta_t_rad")
        if not np.sum(np.isnan(df_subj["e_t_rad"])) == 0:
            sys.exit("Unexpected NaNs in e_t_rad")
        if not np.sum(np.isnan(df_subj["a_t_rad"])) == 8:
            sys.exit("Unexpected NaNs in a_t_rad")
        if not np.sum(np.isnan(df_subj["b_t_rad"])) == 0:
            sys.exit("Unexpected NaNs in b_t_rad")
        if not np.sum(np.isnan(df_subj["group"])) == 0:
            sys.exit("Unexpected NaNs in group")
        if not np.sum(np.isnan(df_subj["kappa_dummy"])) == 0:
            sys.exit("Unexpected NaNs in kappa_dummy")
        if not np.sum(np.isnan(df_subj["sigma"])) == 0:
            sys.exit("Unexpected NaNs in sigma")
        if not np.sum(np.isnan(df_subj["hit_dummy"])) == 0:
            sys.exit("Unexpected NaNs in hit_dummy")
        if not np.sum(np.isnan(df_subj["pers"])) == 0:
            sys.exit("Unexpected NaNs in pers")
        if not np.sum(np.isnan(df_subj["group"])) == 0:
            sys.exit("Unexpected NaNs in group")

    return data


# Run preprocessing
# -----------------
data_pn = preprocessing()
data_pn.name = "data_prepr"

# Save data
safe_save_dataframe(data_pn)
