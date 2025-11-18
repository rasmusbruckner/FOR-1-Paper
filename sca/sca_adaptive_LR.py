"""SCA analysis for adaptive learning rate."""

if __name__ == "__main__":

    import os
    import platform

    import matplotlib

    # Simple cross-platform backend selection
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")  # Headless
    elif platform.system() == "Darwin":
        matplotlib.use("MacOSX")  # macOS native
    else:
        # matplotlib.use("QtAgg")  # Linux with display, Windows, others
        matplotlib.use("Qt5Agg")

    import matplotlib.pyplot as plt

    # Enable interactive mode for debugging
    plt.ion()

    import os

    import numpy as np
    import pandas as pd
    from sca_utils import (build_specs_with_vars, check_fa_pool, fa_candidates,
                           plot_sca, run_or_load_model, run_sca,
                           run_sca_permutation_test, sca_fast_model_comp)

    from FOR_1_Paper.regression.ForRegVars import RegVars

    # ------------
    # 1. Load data
    # ------------

    df_for = pd.read_pickle("for_data/data_prepr_model.pkl")
    df_for = df_for.dropna(subset=["delta_t_rad", "a_t_rad"]).reset_index()  # drop nans
    n_subj = len(np.unique(df_for["subj_num"]))  # number of subjects

    # Which factor
    which_factor = pd.read_pickle("for_data/which_factor.pkl")

    # -------------------------
    # Regression specifications
    # -------------------------

    reg_vars = RegVars()
    reg_vars.n_subj = n_subj  # number of subjects
    reg_vars.n_ker = 4  # number of kernels for estimation
    reg_vars.n_sp = 50  # number of random starting points
    reg_vars.rand_sp = True  # use random starting points
    reg_vars.use_prior = False

    regression_specs = {
        "regression_21": {
            "beta_0": True,  # intercept
            "beta_1": True,  # delta_t
            "beta_2": False,  # omega_t
            "beta_3": False,  # tau_t
            "beta_4": True,  # alpha_t
            "beta_5": False,  # r_t
            "beta_6": False,  # sigma_t
            "beta_7": False,  # catch-trial * PE
            "beta_8": False,  # catch-trial * EE
            "omikron_0": True,  # motor noise
            "omikron_1": True,  # learning-rate noise
            "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
            "lambda_1": False,  # perseveration slope (when combined with lambda_1)
        },
        "regression_22": {
            "beta_0": True,  # intercept
            "beta_1": True,  # delta_t
            "beta_2": False,  # omega_t
            "beta_3": False,  # tau_t
            "beta_4": True,  # alpha_t
            "beta_5": True,  # r_t
            "beta_6": False,  # sigma_t
            "beta_7": False,  # catch-trial * PE
            "beta_8": False,  # catch-trial * EE
            "omikron_0": True,  # motor noise
            "omikron_1": True,  # learning-rate noise
            "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
            "lambda_1": False,  # perseveration slope (when combined with lambda_1)
        },
        "regression_23": {
            "beta_0": True,  # intercept
            "beta_1": True,  # delta_t
            "beta_2": False,  # omega_t
            "beta_3": False,  # tau_t
            "beta_4": True,  # alpha_t
            "beta_5": True,  # r_t
            "beta_6": True,  # sigma_t
            "beta_7": False,  # catch-trial * PE
            "beta_8": False,  # catch-trial * EE
            "omikron_0": True,  # motor noise
            "omikron_1": True,  # learning-rate noise
            "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
            "lambda_1": False,  # perseveration slope (when combined with lambda_1)
        },
        "regression_24": {
            "beta_0": True,  # intercept
            "beta_1": True,  # delta_t
            "beta_2": False,  # omega_t
            "beta_3": False,  # tau_t
            "beta_4": True,  # alpha_t
            "beta_5": True,  # r_t
            "beta_6": True,  # sigma_t
            "beta_7": True,  # catch-trial * PE
            "beta_8": False,  # catch-trial * EE
            "omikron_0": True,  # motor noise
            "omikron_1": True,  # learning-rate noise
            "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
            "lambda_1": False,  # perseveration slope (when combined with lambda_1)
        },
    }

    # Initialize variable
    BIC = list()
    fixed_LR = list()
    all_LRs = np.full([n_subj, len(regression_specs.items())], np.nan)

    # Run or load models
    for i, (name, spec) in enumerate(regression_specs.items()):
        result = run_or_load_model(name, spec, reg_vars, df_for)

        # Store results for model comparison
        BIC.append(sum(result["BIC"]))
        fixed_LR.append(np.mean(result["beta_4"]))
        all_LRs[:, i] = result["beta_4"]

    # Fast model comparison
    sca_fast_model_comp(BIC, fixed_LR, all_LRs)

    # ------------------------------
    # Factor analysis specifications
    # ------------------------------

    # Parameter space of factor analysis
    param_space = {
        "analysis_type": ["simple", "bifactor"],
        "data_type": ["big_data"],
        "n_factors": [2, 3, 4, 5],
        "rotation": ["oblimin", "varimax"],  # "cluster",
        "factor_method": ["minres", "ml"],
        "threshold_loadings": [True],
        "threshold_value": [0.2],
        "fs_method": ["Anderson", "Bartlett"],
        "fs_impute": ["mean"],
        "n_observations": [633],
        "n_variables": [118],
    }

    # Analysis specifications based on our parameter space
    pool = list(fa_candidates(param_space))

    # Check if our files in pool match the files from R (where FA was done)
    # ---------------------------------------------------------------------

    # We don't use "cluster", so don't expect to have all files in our dataset
    check_fa_pool(pool, all_expected=False)

    # Define variable rules for the factor analysis
    # ---------------------------------------------

    var_rule_ids = ["psychosis"]
    var_rules = {
        "psychosis": [
            lambda fa: True,  # always applies
        ],
    }

    # var_rule_ids = ["g_only"]
    # var_rules = {
    #     "g_only": [
    #         lambda fa: fa["analysis_type"] == "bifactor"
    #     ],
    # }

    # Create analysis specifications based on pool and variable rules
    analysis_specs = build_specs_with_vars(pool, var_rule_ids, var_rules)

    # --------------------------------
    # Run specification curve analysis
    # --------------------------------

    # Ensure that we have an empty folder w/o any previous results
    os.makedirs("for_data/sca", exist_ok=True)
    for f in os.listdir("for_data/sca/"):
        os.remove(os.path.join("for_data/sca/", f))

    # Evaluate all specifications
    all_factors, all_betas, all_results = run_sca(
        regression_specs,
        analysis_specs,
        reg_vars,
        df_for,
        which_factor,
        which_var="beta_4",
    )

    # ----------------
    # Permutation test
    # ----------------

    p_T1 = run_sca_permutation_test(all_factors, all_betas, all_results)

    # --------
    # Plotting
    # --------

    plot_sca(var_rule_ids, p_T1, ylabel="Effect Size Adaptive Learning Rate")

    plt.ioff()
    plt.show()
