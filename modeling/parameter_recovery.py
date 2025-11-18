"""Parameter recovery.

1. Load data
2. Simulate data for recovery
3. Estimate models
4. Plot correlations
"""

if __name__ == "__main__":

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from allinpy import latex_plt
    from ForEstimation import ForEstimation
    from ForEstVars import ForEstVars
    from rbmpy import AgentVars

    from FOR_1_Paper.for_utilities import recovery_summary, safe_save_dataframe
    from FOR_1_Paper.modeling.simulation_rbm import simulation_loop

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        pass

    # Turn interactive mode on
    plt.ion()

    # Set random number generator for reproducible results
    np.random.seed(123)

    # ------------
    # 1. Load data
    # ------------

    # Load preprocessed data
    df_exp = pd.read_pickle("for_data/data_prepr.pkl")
    n_subj = len(np.unique(df_exp["subj_num"]))  # number of subjects

    # Load estimated model parameters
    df_estimates = pd.read_pickle("for_data/for_estimates_10_sp.pkl")

    # -----------------------------
    # 2. Simulate data for recovery
    # -----------------------------

    # Call AlEstVars object
    est_vars = ForEstVars()
    est_vars.n_subj = n_subj  # number of subjects
    est_vars.n_ker = 4  # number of kernels for estimation
    est_vars.n_sp = 5  # number of random starting points
    est_vars.rand_sp = True  # use random starting points
    est_vars.use_prior = (
        False  # use weakly informative prior for uncertainty underestimation
    )
    est_vars.circular = True

    # ------------------
    # 3. Estimate models
    # ------------------

    # Fixed values for fixed parameters
    est_vars.fixed_mod_coeffs = {
        est_vars.omikron_0: 5.0,
        est_vars.omikron_1: 0.0,
        est_vars.h: 0.1,
        est_vars.s: 1.0,
        est_vars.u: 0.0,
        est_vars.sigma_H: 0.0001,
    }

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

    # Simulation parameters
    df_model = pd.DataFrame(
        columns=["omikron_0", "omikron_1", "h", "s", "u", "sigma_H", "subj_num"]
    )

    use_subject_estimates = False
    if est_vars.which_vars["omikron_0"]:
        if use_subject_estimates:
            df_model.loc[:, "omikron_0"] = df_estimates["omikron_0"].to_numpy()
        else:
            df_model.loc[:, "omikron_0"] = np.random.uniform(
                low=5.0, high=10.0, size=n_subj
            )
    else:
        df_model.loc[:, "omikron_0"] = np.repeat(
            est_vars.fixed_mod_coeffs["omikron_0"], n_subj
        )

    if est_vars.which_vars["omikron_1"]:
        if use_subject_estimates:
            df_model.loc[:, "omikron_1"] = df_estimates["omikron_1"].to_numpy()
        else:
            df_model.loc[:, "omikron_1"] = np.random.uniform(
                low=0.0, high=0.3, size=n_subj
            )
    else:
        df_model.loc[:, "omikron_1"] = np.repeat(
            est_vars.fixed_mod_coeffs["omikron_1"], n_subj
        )

    if est_vars.which_vars["h"]:
        if use_subject_estimates:
            df_model.loc[:, "h"] = df_estimates["h"].to_numpy()
        else:
            df_model.loc[:, "h"] = np.random.uniform(low=0.0, high=1.0, size=n_subj)
    else:
        df_model.loc[:, "h"] = np.repeat(est_vars.fixed_mod_coeffs["h"], n_subj)

    if est_vars.which_vars["s"]:
        if use_subject_estimates:
            df_model.loc[:, "s"] = df_estimates["s"].to_numpy()
        else:
            df_model.loc[:, "s"] = np.random.uniform(low=0.0, high=1.0, size=n_subj)
    else:
        df_model.loc[:, "s"] = np.repeat(est_vars.fixed_mod_coeffs["s"], n_subj)

    if est_vars.which_vars["u"]:
        if use_subject_estimates:
            df_model.loc[:, "u"] = df_estimates["u"].to_numpy()
        else:
            df_model.loc[:, "u"] = np.random.uniform(low=1.0, high=5.0, size=n_subj)
    else:
        df_model.loc[:, "u"] = np.repeat(est_vars.fixed_mod_coeffs["u"], n_subj)

    if est_vars.which_vars["sigma_H"]:
        if use_subject_estimates:
            df_model.loc[:, "sigma_H"] = df_estimates["sigma_H"].to_numpy()
        else:
            df_model.loc[:, "sigma_H"] = np.random.uniform(
                low=0.0, high=0.5, size=n_subj
            )
    else:
        df_model.loc[:, "sigma_H"] = np.repeat(
            est_vars.fixed_mod_coeffs["sigma_H"], n_subj
        )

    df_model.loc[:, "subj_num"] = df_estimates["subj_num"].to_numpy()

    n_sim = 1  # 1 simulation per subject
    sim_pers = False  # no perseveration
    all_est_errs, df_sim = simulation_loop(
        df_exp, df_model, n_subj, plot_data=False, n_sim=n_sim, sim=True
    )

    df_recov = pd.DataFrame(index=range(0, len(df_sim)), dtype="float")
    df_recov["subj_num"] = df_exp["subj_num"].copy()
    df_recov["new_block"] = df_exp["new_block"].copy()
    df_recov["x_t_rad"] = df_exp["x_t_rad"].copy()
    df_recov["a_t_rad"] = df_sim["sim_a_t_rad"].copy()
    df_recov["delta_t_rad"] = df_sim["delta_t_rad"].copy()
    df_recov["v_t"] = df_exp["v_t"].copy()
    df_recov["sigma"] = df_exp["sigma"].copy()
    df_recov["mu_t_rad"] = df_exp["mu_t_rad"].copy()
    df_recov["b_t_rad"] = df_sim["sim_b_t_rad"].copy()

    # Call AgentVars Object
    agent_vars = AgentVars()
    agent_vars.max_x = 2 * np.pi

    # Estimate parameters and save data
    results_df = al_estimation.parallel_estimation(df_recov, agent_vars)
    results_df.name = "parameter_recovery_" + str(est_vars.n_sp) + "_sp"
    safe_save_dataframe(results_df)

    # --------------------
    # 4. Plot correlations
    # --------------------

    behav_labels = [
        "omikron_0",
        "omikron_1",
        "h",
        "s",
        "u",
        "sigma_H",
    ]

    # Filter based on estimated parameters
    which_params_vec = list(est_vars.which_vars.values())
    behav_labels = [label for label, use in zip(behav_labels, which_params_vec) if use]

    grid_size = (3, 4)
    recovery_summary(df_model, results_df, behav_labels, grid_size)
    plt.ioff()
    plt.show()
