"""FOR circular regression: This script runs circular regression analyses on the FOR dateset.

1. Load data
2. Large model space
3. Model comparison based on BIC
4. Fit winning model separately to low- and high-noise conditions
"""

if __name__ == "__main__":
    import os

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from all_in import latex_plt
    from ForRegVars import RegVars
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
    df_for = df_for.dropna(subset=["delta_t_rad", "a_t_rad"]).reset_index()  # drop nans
    n_subj = len(np.unique(df_for["subj_num"]))  # number of subjects

    # Define regression variables
    # ---------------------------

    reg_vars = RegVars()
    reg_vars.n_subj = n_subj  # number of subjects
    reg_vars.n_ker = 4  # number of kernels for estimation
    reg_vars.n_sp = 50  # 50  # number of random starting points
    reg_vars.rand_sp = True  # use random starting points
    reg_vars.use_prior = False

    # --------------------
    # 2. Large model space
    # --------------------

    # 1.1) Fixed learning rate
    # ------------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: False,  # omega_t
        reg_vars.beta_3: False,  # tau_t
        reg_vars.beta_4: False,  # alpha_t
        reg_vars.beta_5: False,  # r_t
        reg_vars.beta_6: False,  # sigma_t
        reg_vars.beta_7: False,  # catch-trial * PE
        reg_vars.beta_8: False,  # catch-trial * EE
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
        reg_vars.beta_8,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    for_regression = RegressionFor(reg_vars)  # initialize regression object
    model_1_1 = for_regression.parallel_estimation(
        df_for, prior_columns
    )  # run regression
    model_1_1.name = "regression_model_1_1_" + str(reg_vars.n_sp) + "_sp"  # save data
    safe_save_dataframe(model_1_1)

    # 1.2) Fixed LR + omega_t
    # ---------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: True,  # omega_t
        reg_vars.beta_3: False,  # tau_t
        reg_vars.beta_4: False,  # alpha_t
        reg_vars.beta_5: False,  # r_t
        reg_vars.beta_6: False,  # sigma_t
        reg_vars.beta_7: False,  # catch-trial * PE
        reg_vars.beta_8: False,  # catch-trial * EE
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
        reg_vars.beta_8,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    for_regression = RegressionFor(reg_vars)  # initialize regression object
    model_1_2 = for_regression.parallel_estimation(
        df_for, prior_columns
    )  # run regression
    model_1_2.name = "regression_model_1_2_" + str(reg_vars.n_sp) + "_sp"  # save data
    safe_save_dataframe(model_1_2)

    # 1.3) Fixed LR + omega + tau
    # -------------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: True,  # omega_t
        reg_vars.beta_3: True,  # tau_t
        reg_vars.beta_4: False,  # alpha_t
        reg_vars.beta_5: False,  # r_t
        reg_vars.beta_6: False,  # sigma_t
        reg_vars.beta_7: False,  # catch-trial * PE
        reg_vars.beta_8: False,  # catch-trial * EE
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
        reg_vars.beta_8,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    for_regression = RegressionFor(reg_vars)  # initialize regression object
    model_1_3 = for_regression.parallel_estimation(
        df_for, prior_columns
    )  # run regression
    model_1_3.name = "regression_model_1_3_" + str(reg_vars.n_sp) + "_sp"  # save data
    safe_save_dataframe(model_1_3)

    # 1.4) Fixed LR + omega + tau + hit
    # -------------------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: True,  # omega_t
        reg_vars.beta_3: True,  # tau_t
        reg_vars.beta_4: False,  # alpha_t
        reg_vars.beta_5: True,  # r_t
        reg_vars.beta_6: False,  # sigma_t
        reg_vars.beta_7: False,  # catch-trial * PE
        reg_vars.beta_8: False,  # catch-trial * EE
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
        reg_vars.beta_8,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    for_regression = RegressionFor(reg_vars)  # initialize regression object
    model_1_4 = for_regression.parallel_estimation(
        df_for, prior_columns
    )  # run regression
    model_1_4.name = "regression_model_1_4_" + str(reg_vars.n_sp) + "_sp"  # save data
    safe_save_dataframe(model_1_4)

    # 1.5) Fixed LR + omega + tau + hit + noise
    # ---------------------------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: True,  # omega_t
        reg_vars.beta_3: True,  # tau_t
        reg_vars.beta_4: False,  # alpha_t
        reg_vars.beta_5: True,  # r_t
        reg_vars.beta_6: True,  # sigma_t
        reg_vars.beta_7: False,  # catch-trial * PE
        reg_vars.beta_8: False,  # catch-trial * EE
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
        reg_vars.beta_8,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    for_regression = RegressionFor(reg_vars)  # initialize regression object
    model_1_5 = for_regression.parallel_estimation(
        df_for, prior_columns
    )  # run regression
    model_1_5.name = "regression_model_1_5_" + str(reg_vars.n_sp) + "_sp"  # save data
    safe_save_dataframe(model_1_5)

    # 1.6) Fixed LR + omega + tau + hit + noise + catch trial ("full model")
    # --------------------------------------------------------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: True,  # omega_t
        reg_vars.beta_3: True,  # tau_t
        reg_vars.beta_4: False,  # alpha_t
        reg_vars.beta_5: True,  # r_t
        reg_vars.beta_6: True,  # sigma_t
        reg_vars.beta_7: True,  # catch-trial * PE
        reg_vars.beta_8: False,  # catch-trial * EE
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
        reg_vars.beta_8,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    for_regression = RegressionFor(reg_vars)  # initialize regression object
    model_1_6 = for_regression.parallel_estimation(
        df_for, prior_columns
    )  # run regression
    model_1_6.name = "regression_model_1_6_" + str(reg_vars.n_sp) + "_sp"  # save data
    safe_save_dataframe(model_1_6)

    # 2.1) Fixed LR + adaptive LR
    # ---------------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: False,  # omega_t
        reg_vars.beta_3: False,  # tau_t
        reg_vars.beta_4: True,  # alpha_t
        reg_vars.beta_5: False,  # r_t
        reg_vars.beta_6: False,  # sigma_t
        reg_vars.beta_7: False,  # catch-trial * PE
        reg_vars.beta_8: False,  # catch-trial * EE
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
        reg_vars.beta_8,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    for_regression = RegressionFor(reg_vars)  # initialize regression object
    model_2_1 = for_regression.parallel_estimation(
        df_for, prior_columns
    )  # run regression
    model_2_1.name = "regression_model_2_1_" + str(reg_vars.n_sp) + "_sp"  # save data
    safe_save_dataframe(model_2_1)

    # 2.2) Fixed LR + adaptive LR + hit
    # ---------------------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: False,  # omega_t
        reg_vars.beta_3: False,  # tau_t
        reg_vars.beta_4: True,  # alpha_t
        reg_vars.beta_5: True,  # r_t
        reg_vars.beta_6: False,  # sigma_t
        reg_vars.beta_7: False,  # catch-trial * PE
        reg_vars.beta_8: False,  # catch-trial * EE
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
        reg_vars.beta_8,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    for_regression = RegressionFor(reg_vars)  # initialize regression object
    model_2_2 = for_regression.parallel_estimation(
        df_for, prior_columns
    )  # run regression
    model_2_2.name = "regression_model_2_2_" + str(reg_vars.n_sp) + "_sp"  # save data
    safe_save_dataframe(model_2_2)

    # 2.3) Fixed LR + adaptive LR + hit +  noise
    # ------------------------------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: False,  # omega_t
        reg_vars.beta_3: False,  # tau_t
        reg_vars.beta_4: True,  # alpha_t
        reg_vars.beta_5: True,  # r_t
        reg_vars.beta_6: True,  # sigma_t
        reg_vars.beta_7: False,  # catch-trial * PE
        reg_vars.beta_8: False,  # catch-trial * EE
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
        reg_vars.beta_8,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    for_regression = RegressionFor(reg_vars)  # initialize regression object
    model_2_3 = for_regression.parallel_estimation(
        df_for, prior_columns
    )  # run regression
    model_2_3.name = "regression_model_2_3_" + str(reg_vars.n_sp) + "_sp"  # save data
    safe_save_dataframe(model_2_3)

    # 2.4) Fixed LR + adaptive LR + hit + noise + catch trial
    # -------------------------------------------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: False,  # omega_t
        reg_vars.beta_3: False,  # tau_t
        reg_vars.beta_4: True,  # alpha_t
        reg_vars.beta_5: True,  # r_t
        reg_vars.beta_6: True,  # sigma_t
        reg_vars.beta_7: True,  # catch-trial * PE
        reg_vars.beta_8: False,  # catch-trial * EE
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
        reg_vars.beta_8,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    for_regression = RegressionFor(reg_vars)  # initialize regression object
    model_2_4 = for_regression.parallel_estimation(
        df_for, prior_columns
    )  # run regression
    model_2_4.name = "regression_model_2_4_" + str(reg_vars.n_sp) + "_sp"  # save data
    safe_save_dataframe(model_2_4)

    # --------------------------------
    # 3. Model comparison based on BIC
    # --------------------------------

    plt.figure()
    bic_values = [
        sum(model_1_1["BIC"]),
        sum(model_1_2["BIC"]),
        sum(model_1_3["BIC"]),
        sum(model_1_4["BIC"]),
        sum(model_1_5["BIC"]),
        sum(model_1_6["BIC"]),
        sum(model_2_1["BIC"]),
        sum(model_2_2["BIC"]),
        sum(model_2_3["BIC"]),
        sum(model_2_4["BIC"]),
    ]
    plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], bic_values)
    plt.xticks(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [
            "Model 1.1",
            "Model 1.2",
            "Model 1.3",
            "Model 1.4",
            "Model 1.5",
            "Model 1.6",
            "Model 2.1",
            "Model 2.2",
            "Model 2.3",
            "Model 2.4",
        ],
    )
    plt.ylabel("Sum BIC")
    sns.despine()

    print(
        "Best model: ",
        bic_values.index(max(bic_values)) + 1,
        "with BIC = ",
        max(bic_values),
    )  # +1 because starts from 0

    # -----------------------------------------------------------------
    # 4. Fit winning model separately to low- and high-noise conditions
    # -----------------------------------------------------------------

    df_for_low_noise = df_for[df_for["kappa_t"] == 16].copy()
    df_for_high_noise = df_for[df_for["kappa_t"] == 8].copy()

    # 3) Model 2.3: Fixed LR + adaptive LR + hit + noise
    # --------------------------------------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: False,  # omega_t
        reg_vars.beta_3: False,  # tau_t
        reg_vars.beta_4: True,  # alpha_t
        reg_vars.beta_5: True,  # r_t
        reg_vars.beta_6: True,  # sigma_t
        reg_vars.beta_7: False,  # catch-trial * PE
        reg_vars.beta_8: False,  # catch-trial * EE
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
        reg_vars.beta_8,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    # Low noise
    for_regression = RegressionFor(reg_vars)  # initialize regression object
    model_2_3 = for_regression.parallel_estimation(
        df_for_low_noise, prior_columns
    )  # run regression
    model_2_3.name = (
        "regression_model_low_noise_2_3_" + str(reg_vars.n_sp) + "_sp"
    )  # save data
    safe_save_dataframe(model_2_3)

    # High noise
    for_regression = RegressionFor(reg_vars)  # initialize regression object
    model_2_3 = for_regression.parallel_estimation(
        df_for_high_noise, prior_columns
    )  # run regression
    model_2_3.name = (
        "regression_model_high_noise_2_3_" + str(reg_vars.n_sp) + "_sp"
    )  # save data
    safe_save_dataframe(model_2_3)

    # Show plot
    plt.ioff()
    plt.show()
