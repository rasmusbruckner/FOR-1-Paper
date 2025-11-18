"""Performs the model estimation."""

if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from ForEstimation import ForEstimation
    from ForEstVars import ForEstVars
    from rbmpy import AgentVars

    from FOR_1_Paper.for_utilities import safe_save_dataframe

    # Turn interactive mode on
    plt.ion()

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # ------------
    # 1. Load data
    # ------------

    df_for = pd.read_pickle("for_data/data_prepr.pkl")
    n_subj = len(np.unique(df_for["subj_num"]))

    # -------------------
    # 2. Prepare analysis
    # -------------------

    # Call AgentVars Object
    agent_vars = AgentVars()
    agent_vars.max_x = 2 * np.pi

    # Call AlEstVars object
    est_vars = ForEstVars()
    est_vars.n_subj = n_subj  # number of subjects
    est_vars.n_ker = 4  # number of kernels for estimation
    est_vars.n_sp = 10  # number of random starting points
    est_vars.rand_sp = True  # use random starting points
    est_vars.use_prior = (
        True  # use weakly informative prior for uncertainty underestimation
    )

    # -----------------
    # 3. Estimate model
    # -----------------

    # Free parameters
    est_vars.which_vars = {
        est_vars.omikron_0: True,  # motor noise
        est_vars.omikron_1: True,  # learning-rate noise
        est_vars.h: True,  # hazard rate
        est_vars.s: True,  # surprise sensitivity
        est_vars.u: True,  # uncertainty underestimation
        est_vars.sigma_H: True,  # catch trials
    }

    # Call AlEstimation object
    al_estimation = ForEstimation(est_vars)

    # Estimate parameters and save data
    results_df = al_estimation.parallel_estimation(df_for, agent_vars)

    results_df.name = "for_estimates_" + str(est_vars.n_sp) + "_sp"
    safe_save_dataframe(results_df)
