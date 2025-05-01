"""Parameter recovery

1. Load data

"""

# todo: in anderen branch und erstmal ignorieren WIP!

if __name__ == "__main__":

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from all_in import get_df_subj, latex_plt
    from ForEstimation import ForEstimation
    from ForEstVars import ForEstVars
    from rbm_analyses import AgentVars, AlAgent

    from FOR_1_Paper.for_modeling.for_simulation_rbm import simulation_loop
    from FOR_1_Paper.for_modeling.for_task_agent_int_rbm import task_agent_int
    from FOR_1_Paper.for_utilities import recovery_summary, safe_save_dataframe

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
    # todo: nochmal neu fitten mit 15 sp wegen catch trial
    df_estimates = pd.read_pickle("for_data/for_estimates_5_sp.pkl")

    # # Simulation parameters
    # model = pd.DataFrame(
    #     columns=["omikron_0", "omikron_1", "h", "s", "u", "sigma_H", "subj_num"]
    # )
    # model.loc[:, "omikron_0"] = df_estimates["omikron_0"].to_numpy()
    # model.loc[:, "omikron_1"] = df_estimates["omikron_1"].to_numpy()
    # model.loc[:, "h"] = df_estimates["h"].to_numpy()
    # model.loc[:, "s"] = df_estimates["s"].to_numpy()
    # model.loc[:, "u"] = df_estimates["u"].to_numpy()
    # model.loc[:, "sigma_H"] = df_estimates["sigma_H"].to_numpy()
    # model.loc[:, "subj_num"] = df_estimates["subj_num"].to_numpy()

    # df_exp['v_t'] = 0
    # df_exp['v_t'] = np.random.randint(0,2, size=len(df_exp['v_t']))
    # df_exp['v_t'] = np.random.binomial(n=1, p=0.5, size=len(df_exp['v_t']))
    # Simulation parameters
    model = pd.DataFrame(
        columns=["omikron_0", "omikron_1", "h", "s", "u", "sigma_H", "subj_num"]
    )
    model.loc[:, "omikron_0"] = df_estimates["omikron_0"].to_numpy()
    # model.loc[:, "omikron_0"] = 1#5 #np.random.rand(len(df_estimates["omikron_0"]))*9 + 1#df_estimates["omikron_0"].to_numpy()
    # model.loc[:, "omikron_0"] = 0.1
    model.loc[:, "omikron_1"] = df_estimates["omikron_1"].to_numpy()  # 0.0 #
    model.loc[:, "h"] = df_estimates["h"].to_numpy()
    model.loc[:, "s"] = df_estimates["s"].to_numpy()
    model.loc[:, "u"] = df_estimates["u"].to_numpy()
    model.loc[:, "sigma_H"] = (
        np.random.rand(len(df_estimates["omikron_0"])) * 0.6 + 0.001
    )  # df_estimates["sigma_H"].to_numpy()
    model.loc[:, "subj_num"] = df_estimates["subj_num"].to_numpy()

    model.loc[0, "sigma_H"] = 0.25

    # -----------------------------------
    # 2. Simulate data for recovery model
    # -----------------------------------

    n_sim = 1  # 1 simulation per subject
    sim_pers = False  # no perseveration
    all_est_errs, all_data = simulation_loop(
        df_exp, model, n_subj, plot_data=False, n_sim=n_sim, sim=True
    )

    # ensure that model is aware of actual heli pos on catch trials!! hier war ein fehlen
    all_data["mu_t_rad"] = df_exp["mu_t_rad"].copy()

    # Agent-variables object
    agent_vars = AgentVars()
    agent_vars.max_x = 2 * np.pi
    agent_vars.mu_0 = 0

    # For consistency with Matt
    agent_vars.tau_0 = 0.99
    agent_vars.sigma_0 = (
        6.1875  # due to his initRU settings that I recomputed to sigma_0
    )

    agent_vars.circular = True

    # Agent object
    agent = AlAgent(agent_vars)

    sel_coeffs = model[model["subj_num"] == 1].values.flatten().tolist()

    # Extract subject-specific data frame
    df_subj = get_df_subj(all_data, 0)

    # todo: hier muss sim_b_t in b_t_rad umgenannt werden... bzw. nicht umbenannt sondern wie in recov neuer df eig.
    # df_exp_s1 = get_df_subj(df_exp, 0)

    llh_sim = -1 * np.nansum(df_subj["sim_a_t_llh"])
    llh_sim_vec = -1 * df_subj["sim_a_t_llh"]

    # Run task-agent interaction
    sum_llh_rbm = list()
    sum_llh_sim = list()

    sim = False
    sigma_H_vals = np.linspace(0.001, 0.5, 100)

    agent_vars.circular = True
    for i in range(len(sigma_H_vals)):
        sel_coeffs[5] = sigma_H_vals[i]

        # Set agent variables of current participant
        agent_vars.h = sel_coeffs[2]
        agent_vars.s = sel_coeffs[3]
        agent_vars.u = np.exp(sel_coeffs[4])
        agent_vars.sigma_H = sel_coeffs[5]  # model.loc[0, "sigma_H"] # #  # #

        # Agent object
        agent = AlAgent(agent_vars)

        llh_rbm, sim_data = task_agent_int(
            df_subj, agent, agent_vars, sel_coeffs, sim=sim
        )
        sum_llh_rbm.append(-1 * llh_rbm.sum())
        llh_hää = -1 * llh_rbm

        # sum_llh_sim.append(-1 * sim_data['a_t_rad_llh'].sum())
        # todo: frage der fragen: warum llh schlechter bei richtigem param obwohl a_t etc. eig. gleich??
        # a_t_hat vergleichen und ob wirklich richtiges a_t_rad genommen

    plt.figure()
    plt.plot(sigma_H_vals, sum_llh_rbm)
    # plt.ioff()
    print(
        "minimum:",
        sigma_H_vals[sum_llh_rbm == min(sum_llh_rbm)],
        "actual:",
        model.loc[0, "sigma_H"],
    )
    print("min llh:", min(sum_llh_rbm), "llh_sim:", llh_sim)

    # plt.show()
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
    est_vars.n_sp = 1  # 10  # number of random starting points
    est_vars.rand_sp = False  # True  # use random starting points
    est_vars.use_prior = (
        True  # use weakly informative prior for uncertainty underestimation
    )
    est_vars.fixed_mod_coeffs = {
        "omikron_0": 1.0,
        "omikron_1": 0.0,
        "h": 0.9,
        "s": 0.1,
        "u": 0.0,
        "sigma_H": 0.0001,
    }
    # todo:
    #   add recovery for agent here
    #   and simple example in package with only noise and haz or so

    # ------------------
    # 3. Estimate models
    # ------------------

    # Free parameters
    est_vars.which_vars = {
        est_vars.omikron_0: True,  # motor noise
        est_vars.omikron_1: True,  # False,  # learning-rate noise
        est_vars.h: True,  # hazard rate
        est_vars.s: True,  # surprise sensitivity
        est_vars.u: True,  # uncertainty underestimation
        est_vars.sigma_H: True,  # catch trials
    }

    # Specify that experiment 1 is modeled
    est_vars.which_exp = 1

    est_vars.circular = True

    # Call AlEstimation object
    al_estimation = ForEstimation(est_vars)

    # Estimate parameters and save data
    results_df = al_estimation.parallel_estimation(all_data, agent_vars)

    results_df.name = "parameter_recovery_" + str(est_vars.n_sp) + "_sp"
    safe_save_dataframe(results_df)

    # --------------------
    # 3. Plot correlations
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
    recovery_summary(model, results_df, behav_labels, grid_size)
    plt.ioff()
    plt.show()
