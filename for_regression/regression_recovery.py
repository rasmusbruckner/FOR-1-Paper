"""Parameter recovery regression model.

1. Simulate data for recovery
2. Estimate regression model
3. Plot correlations
"""

if __name__ == "__main__":

    import os

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from all_in import latex_plt
    from ForRegVars import RegVars
    from RegressionFor import RegressionFor

    from FOR_1_Paper.for_utilities import recovery_summary, safe_save_dataframe

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    # Use preferred backend for Linux, or just take default
    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        pass

    # Turn interactive mode on
    plt.ion()

    # Set random number generator for reproducible results
    np.random.seed(123)

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # Number of random starting points for regression estimation
    n_sp = 10

    # Load data
    df_for = pd.read_pickle("for_data/data_prepr_model.pkl")
    n_subj = len(np.unique(df_for["subj_num"]))

    # -----------------------------
    # 1. Simulate data for recovery
    # -----------------------------

    # Initialize regression variables
    reg_vars = RegVars()
    reg_vars.n_subj = n_subj
    reg_vars.n_sp = n_sp
    reg_vars.n_ker = 4
    reg_vars.use_prior = True
    reg_vars.rand_sp = True

    # Determine which parameters should be estimated
    reg_vars.which_vars = {
        "beta_0": True,  # Intercept
        "beta_1": True,  # PE (fixed learning rate)
        "beta_2": True,  # Interaction PE and RU
        "beta_3": True,  # Interaction PE and CPP
        "beta_4": True,  # Interaction PE and hit
        "beta_5": True,  # Interaction PE and noise condition
        "beta_6": True,  # Interaction PE and visible
        "beta_7": False,  # Interaction EE and visible
        "omikron_0": True,  # Motor noise (independent of UP)
        "omikron_1": True,  # Learning-rate noise (dependent on UP)
        "lambda_0": False,  # Perseveration intercept
        "lambda_1": False,  # Perseveration slope
    }

    # When lambda_1 involved, adjust estimation variables accordingly
    if (
        reg_vars.which_vars["lambda_0"] and reg_vars.which_vars["lambda_1"]
    ) or reg_vars.which_vars["lambda_1"]:

        # Boundaries
        reg_vars.lambda_0_bnds = (-3, 3)
        reg_vars.lambda_1_bnds = (-5, 0.0)

        # Starting-point range
        reg_vars.lambda_0_x0_range = (-1, 0)
        reg_vars.lambda_1_x0_range = (-5, 0.0)

        # Starting point
        reg_vars.lambda_0_x0 = 0
        reg_vars.lambda_1_x0 = -1

        # Combined bounds
        reg_vars.bnds = [
            reg_vars.beta_0_bnds,
            reg_vars.beta_1_bnds,
            reg_vars.beta_2_bnds,
            reg_vars.beta_3_bnds,
            reg_vars.beta_4_bnds,
            reg_vars.beta_5_bnds,
            reg_vars.beta_6_bnds,
            reg_vars.beta_7_bnds,
            reg_vars.omikron_0_bnds,
            reg_vars.omikron_1_bnds,
            reg_vars.lambda_0_bnds,
            reg_vars.lambda_1_bnds,
        ]

    # Create regression-components list
    reg_vars.regressionComponents = [
        reg_vars.which_vars["beta_0"],
        reg_vars.which_vars["beta_1"],
        reg_vars.which_vars["beta_2"],
        reg_vars.which_vars["beta_3"],
        reg_vars.which_vars["beta_4"],
        reg_vars.which_vars["beta_5"],
        reg_vars.which_vars["beta_6"],
        reg_vars.which_vars["beta_7"],
    ]

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

    # Create regression-object instance
    regression = RegressionFor(reg_vars)

    # Sample random model parameters that we try to recover
    df_params = pd.DataFrame()

    if reg_vars.which_vars["beta_0"]:
        df_params["beta_0"] = np.random.uniform(-0.5, 0.5, size=n_subj)

    if reg_vars.which_vars["beta_1"]:
        df_params["beta_1"] = np.random.uniform(0.5, 1.0, size=n_subj)

    if reg_vars.which_vars["beta_2"]:
        df_params["beta_2"] = np.random.rand(n_subj)

    if reg_vars.which_vars["beta_3"]:
        df_params["beta_3"] = np.random.rand(n_subj)

    if reg_vars.which_vars["beta_4"]:
        df_params["beta_4"] = np.random.rand(n_subj)

    if reg_vars.which_vars["beta_5"]:
        df_params["beta_5"] = np.random.uniform(-0.1, 0.1, size=n_subj)

    if reg_vars.which_vars["beta_6"]:
        df_params["beta_6"] = np.random.uniform(-1.0, 0.1, size=n_subj)

    if reg_vars.which_vars["beta_7"]:
        df_params["beta_7"] = np.random.uniform(-0.1, 1.0, size=n_subj)

    if reg_vars.which_vars["omikron_0"]:
        df_params["omikron_0"] = np.random.uniform(1, 15, size=n_subj)

    if reg_vars.which_vars["omikron_1"]:
        df_params["omikron_1"] = np.random.rand(n_subj) * 0.2

    if reg_vars.which_vars["lambda_0"] and not reg_vars.which_vars["lambda_1"]:
        df_params["lambda_0"] = np.random.uniform(0.1, 0.6, size=n_subj)

    if reg_vars.which_vars["lambda_0"] and reg_vars.which_vars["lambda_1"]:
        df_params["lambda_0"] = np.random.uniform(-3, 3, size=n_subj)
        df_params["lambda_1"] = np.random.uniform(-2, -0.1, size=n_subj)

    if not reg_vars.which_vars["lambda_0"] and reg_vars.which_vars["lambda_1"]:
        df_params["lambda_1"] = np.random.uniform(-5, -0.0, size=n_subj)

    df_params["subj_num"] = np.arange(1, n_subj + 1)

    # Simulate updates based on sampled parameters
    n_trials = 400
    samples = regression.sample_data(df_params, n_trials, df_for)

    # ----------------------------
    # 2. Estimate regression model
    # ----------------------------

    # Estimate regression model
    results_df = regression.parallel_estimation(samples, prior_columns)
    results_df.name = "regression_recovery_" + str(reg_vars.n_sp) + "_sp"
    safe_save_dataframe(results_df)

    # --------------------
    # 3. Plot correlations
    # --------------------

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
    recovery_summary(df_params, results_df, behav_labels, grid_size)
    plt.ioff()
    plt.show()
