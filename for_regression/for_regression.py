"""FOR circular regression: This script runs a circular regression on the FOR dateset.

1. Load data
2. Run regression analysis
3. Plot the results
"""

if __name__ == "__main__":
    import os

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from all_in import latex_plt
    from ForRegVars import RegVars
    from rbm_analyses import parameter_summary
    from RegressionFor import RegressionFor

    from FOR_1_Paper.for_utilities import safe_save_dataframe

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    # Use preferred backend for Linux, or just take default
    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        pass

    # Turn interactive mode on
    plt.ion()

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # ------------
    # 1. Load data
    # ------------

    df_for = pd.read_pickle("for_data/data_prepr_model.pkl")
    n_subj = len(np.unique(df_for["subj_num"]))  # number of subjects

    # --------------------------
    # 2. Run regression analysis
    # --------------------------

    # Define regression variables
    # ---------------------------

    reg_vars = RegVars()
    reg_vars.n_subj = n_subj  # number of subjects
    reg_vars.n_ker = 4  # number of kernels for estimation
    reg_vars.n_sp = 10  # 50  # number of random starting points
    reg_vars.rand_sp = True  # use random starting points
    reg_vars.use_prior = True

    # Run regression model
    # --------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: True,  # omega_t
        reg_vars.beta_3: True,  # tau_t
        reg_vars.beta_4: True,  # r_t
        reg_vars.beta_5: True,  # sigma_t
        reg_vars.beta_6: True,  # catch-trial * PE
        reg_vars.beta_7: False,  # catch-trial * EE
        reg_vars.omikron_0: True,  # motor noise
        reg_vars.omikron_1: True,  # learning-rate noise
        reg_vars.lambda_0: False,  # pers intercept when comb w/ lambda_1 or overall probability
        reg_vars.lambda_1: False,  # perseveration slope (when combined with lambda_1)
    }

    # Select parameters according to selected variables and create data frame
    prior_columns = [
        reg_vars.beta_0,
        reg_vars.beta_1,
        reg_vars.beta_2,
        reg_vars.beta_3,
        reg_vars.beta_4,
        reg_vars.beta_5,
        reg_vars.beta_6,
        reg_vars.beta_7,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    # Initialize regression object
    for_regression = RegressionFor(reg_vars)

    # Drop nans
    df_for = df_for.dropna(subset=["delta_t_rad", "a_t_rad"]).reset_index()

    # Run regression
    # --------------

    results_df = for_regression.parallel_estimation(df_for, prior_columns)
    results_df.name = "regression_" + str(reg_vars.n_sp) + "_sp"
    # results_df.name = "regression_full_" + str(reg_vars.n_sp) + "_sp"

    # Save data
    safe_save_dataframe(results_df)

    # ---------------
    # 3. Plot results
    # ---------------

    behav_labels = [
        "beta_0",
        "beta_1",
        "beta_2",
        "beta_3",
        "beta_4",
        "beta_5",
        "beta_6",
        "beta_7",
        "omikron_0",
        "omikron_1",
        "lambda_0",
        "lambda_1",
    ]

    # Filter based on estimated parameters
    which_params_vec = list(reg_vars.which_vars.values())
    behav_labels = [label for label, use in zip(behav_labels, which_params_vec) if use]

    grid_size = (3, 4)
    parameter_summary(results_df, behav_labels, grid_size)

    # Show plot
    plt.ioff()
    plt.show()
