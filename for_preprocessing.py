""" Data Preprocessing: This script performs the preprocessing for the FOR paper """

import numpy as np
from for_utilities import load_data, sorted_nicely, get_file_paths
import pandas as pd
import sys


def preprocessing():
    """ This function loads and preprocesses the adaptive learning BIDS data for further analyses """

    # Data folder
    folder_path = 'for_bids_data'

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
    new_block = data['new_block'].values  # block change indicator

    # Recode last entry in variables of interest of each block to nan
    # to avoid that data of different participants are mixed
    to_nan = np.zeros(len(new_block))
    to_nan[:-1] = new_block[1:]
    to_nan[-1] = 1  # manually added, because no new block after last trial

    # Prediction error:
    data.loc[to_nan == 1, 'delta_t'] = np.nan

    # Absolute estimation error:
    data['e_t'] = abs(data['e_t'])

    # Update: On trial t, we analyze the difference between belief at t and t+1
    a_t = np.full(len(data), np.nan)
    a_t[:-1] = data['a_t'][1:]
    a_t[to_nan == 1] = np.nan
    data['a_t'] = a_t

    # Perseveration: pers := 1, if a_t=0; 0, else
    data['pers'] = a_t == 0

    # Add information on group (here all in the same group)
    data['group'] = 0

    # Rename ID to subj_num for consistency with earlier analyses
    data = data.rename(columns={"ID": "subj_num"})

    # Test if expected values appear in preprocessed data frames
    # ----------------------------------------------------------

    # Extrac number of subjects
    all_id = list(set(data['subj_num']))  # ID for each participant
    n_subj = len(all_id)  # number of participants

    # Cycle over subjects
    for i in range(n_subj):

        # Extract data of current subject
        df_subj = data[(data['subj_num'] == i + 1)].copy()

        # Check expected values
        if not np.sum(np.isnan(df_subj['delta_t'])) == 8:
            sys.exit("Unexpected NaN's in delta_t")
        if not np.sum(np.isnan(df_subj['a_t'])) == 8:
            sys.exit("Unexpected NaN's in a_t")
        if not np.sum(df_subj['new_block']) == 8:
            sys.exit("Unexpected NaN's in new_block")
        if not np.sum(np.isnan(df_subj['subj_num'])) == 0:
            sys.exit("Unexpected NaN's in subj_num")
        if not np.sum(np.isnan(df_subj['group'])) == 0:
            sys.exit("Unexpected NaN's in group")
        if not np.sum(np.isnan(df_subj['x_t'])) == 0:
            sys.exit("Unexpected NaN's in x_t")
        if not np.sum(np.isnan(df_subj['b_t'])) == 0:
            sys.exit("Unexpected NaN's in b_t")
        if not np.sum(np.isnan(df_subj['mu_t'])) == 0:
            sys.exit("Unexpected NaN's in mu_t")
        if not np.sum(np.isnan(df_subj['c_t'])) == 0:
            sys.exit("Unexpected NaN's in c_t")
        if not np.sum(np.isnan(df_subj['r_t'])) == 0:
            sys.exit("Unexpected NaN's in r_t")
        if not np.sum(np.isnan(df_subj['sigma'])) == 0:
            sys.exit("Unexpected NaN's in sigma")
        if not np.sum(np.isnan(df_subj['v_t'])) == 0:
            sys.exit("Unexpected NaN's in v_t")
        if not np.sum(np.isnan(df_subj['e_t'])) == 0:
            sys.exit("Unexpected NaN's in e_t")
        if not np.sum(np.isnan(df_subj['pers'])) == 0:
            sys.exit("Unexpected NaN's in pers")

    return data


# Run preprocessing
# -----------------
data_pn = preprocessing()

# Load previous file for comparison
expected_data_pn = pd.read_pickle('for_bids_data/data_prepr.pkl')

# Test if equal and save data
same = data_pn.equals(expected_data_pn)
print("\nActual and expected preprocessed data equal:", same, "\n")
if not same:
    data_pn.to_pickle('for_bids_data/data_prepr_unexpected.pkl')
else:
    data_pn.to_pickle('for_bids_data/data_prepr.pkl')
