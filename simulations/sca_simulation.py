"""Simulate common latent factor that determines questionnaire
responses and hazard rate for RBM."""

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
        matplotlib.use("Qt5Agg")  # Linux with display, Windows, others

    import matplotlib.pyplot as plt

    # Enable interactive mode for debugging
    plt.ion()

    import random

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from allinpy import cm2inch
    from rbmpy import parameter_summary

    from FOR_1_Paper.for_utilities import plot_correlation, safe_save_dataframe
    from FOR_1_Paper.simulations.sim_utils import (generate_questionnaire_data,
                                                   sample_hazard_rate,
                                                   simulate_and_run_regression)

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # Control random number generator for reproducible results
    seed = 123
    np.random.seed(seed)
    random.seed(seed)

    # Simulation type ("task_agent" or "agent")
    sim = "task_agent"

    # Simulation-type-conditional parameters
    n_subj = None
    n_trials = None
    df_exp = None
    if sim == "agent":
        df_exp = pd.read_pickle("for_data/data_prepr.pkl")
        n_subj = df_exp["subj_num"].nunique()
        n_trials = df_exp["trial"].nunique()
    elif sim == "task_agent":
        n_subj = 200
        n_trials = 400

    data_folder = (
        "for_data/sca_"
        + sim
        + "_N"
        + str(n_subj)
        + "_T"
        + str(n_trials)
        + "_"
        + "seed"
        + str(seed)
        + "/"
    )
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # --------------------------------
    # Simulate questionnaire responses
    # --------------------------------

    questionnaire_data = generate_questionnaire_data(n_subj, theta_sd=0.25, eta_sd=0.25)

    # Reformat data in line with original data from Hamburg
    wide_data = questionnaire_data.pivot(
        index="subj_num",  # subjects become the rows (index)
        columns="item",  # items become the columns
        values="response",  # response values
    )

    # Save sum scores for additional analyses
    sum_scores = (
        questionnaire_data.groupby(["subj_num", "ID", "questionnaire"])
        .sum()["response"]
        .to_frame()
        .reset_index()
    )
    sum_scores_q1 = sum_scores[sum_scores["questionnaire"] == 1].reset_index(drop=True)
    sum_scores_q1.name = (
        "sim_sca_sum_scores_q1_seed"
        + str(seed)
        + "_N"
        + str(n_subj)
        + "_T"
        + str(n_trials)
        + "_"
        + sim
    )
    safe_save_dataframe(sum_scores_q1, data_dir=data_folder)

    sum_scores_q2 = sum_scores[sum_scores["questionnaire"] == 2].reset_index(drop=True)
    sum_scores_q2.name = (
        "sim_sca_sum_scores_q2_seed"
        + str(seed)
        + "_N"
        + str(n_subj)
        + "_T"
        + str(n_trials)
        + "_"
        + sim
    )
    safe_save_dataframe(sum_scores_q2, data_dir=data_folder)

    thetas = (
        questionnaire_data.groupby(["subj_num", "questionnaire"])
        .first()["theta"]
        .to_frame()
        .reset_index()
    )
    thetas = thetas[thetas["questionnaire"] == 1]["theta"].reset_index(drop=True)
    etas = (
        questionnaire_data.groupby(["subj_num", "questionnaire"])
        .first()["eta"]
        .to_frame()
        .reset_index()
    )
    etas = etas[etas["questionnaire"] == 2]["eta"].reset_index(drop=True)

    # Sample hazard rate as a function of thetas
    hazard_rate = sample_hazard_rate(thetas, intercept=0.0, slope=1.5, sigma=1)

    # ------------------------------------------
    # Simulate data using RBM and run regression
    # ------------------------------------------

    if sim == "task_agent":

        kappa = np.tile(np.repeat([16, 8], n_trials / 2), n_subj)
        sigma = np.sqrt(1 / kappa)
        angular_shield_size = 2 * sigma
        subj_num = np.repeat(np.arange(1, n_subj + 1), n_trials)
        new_block = np.tile(np.concatenate([np.array([1]), np.zeros(99)]), 4 * n_subj)
        group = np.tile(np.repeat(1, n_trials), n_subj)
        df_exp = pd.DataFrame(
            {
                "subj_num": subj_num,
                "sigma": sigma,
                "kappa_t": kappa,
                "new_block": new_block,
                "angular_shield_size": angular_shield_size,
                "group": group,
            }
        )

    # Simulate and run regression model
    df_reg, sim_est_errs, df_sim_sca_data, df_rbm = simulate_and_run_regression(
        n_subj, hazard_rate, df_exp, sim=sim
    )

    df_sca_sim = pd.DataFrame(np.arange(1, n_subj + 1), columns=["subj_num"])
    df_sca_sim["hazard_rate"] = hazard_rate
    df_sca_sim["thetas"] = thetas
    df_sca_sim["etas"] = etas
    df_sca_sim["sum_scores_q1"] = sum_scores_q1["response"]
    df_sca_sim["sum_scores_q2"] = sum_scores_q2["response"]
    df_sca_sim["fixed_lr"] = df_reg["beta_1"]
    df_sca_sim["adaptive_lr"] = df_reg["beta_4"]
    df_sca_sim.name = (
            "sim_sca_data_seed"
            + str(seed)
            + "_N"
            + str(n_subj)
            + "_T"
            + str(n_trials)
            + "_"
            + sim
    )
    safe_save_dataframe(df_sca_sim, data_dir=data_folder)

    # Save regression data for SCA
    df_sim_sca_data.name = (
        "sim_sca_regression_seed"
        + str(seed)
        + "_N"
        + str(n_subj)
        + "_T"
        + str(n_trials)
        + "_"
        + sim
    )
    safe_save_dataframe(df_sim_sca_data, data_dir=data_folder)

    # Save RBM data for SCA
    df_rbm.name = (
        "sim_sca_rbm_seed"
        + str(seed)
        + "_N"
        + str(n_subj)
        + "_T"
        + str(n_trials)
        + "_"
        + sim
    )
    safe_save_dataframe(df_rbm, data_dir=data_folder)

    # Plot regression results
    # -----------------------

    behav_labels = [
        "beta_0",
        "beta_1",
        "beta_4",
        "beta_7",
        "omikron_0",
        "omikron_1",
    ]

    axis_labels = [
        "Intercept",
        "Fixed LR",
        "Adaptive LR",
        "Catch trial",
        "Motor Noise",
        "LR Noise",
    ]
    grid_size = (2, 3)
    parameter_summary(df_reg, behav_labels, grid_size, axis_labels=axis_labels)
    plt.savefig("figures/fa_sim_regression", dpi=400)

    # ---------------------------------------------
    # Plot all simulation results for good overview
    # ---------------------------------------------

    # Initialize figure
    plt.figure(figsize=(cm2inch(20, 15)))
    ax_00 = plt.subplot(4, 4, 1)

    # Histograms
    # -----------

    ax_00.hist(questionnaire_data["response"], bins=5, density=True)
    ax_00.set_xlabel("Response")
    ax_00.set_xticks(np.arange(1, 6, 1))
    sns.despine()

    ax_01 = plt.subplot(4, 4, 2)
    ax_01.hist(sum_scores_q1["response"], density=True)
    ax_01.set_xlabel("Sum Scores")
    sns.despine()

    ax_02 = plt.subplot(4, 4, 3)
    ax_02.hist(questionnaire_data["theta"], density=True)
    ax_02.set_xlabel("Latent Factor")
    sns.despine()

    ax_03 = plt.subplot(4, 4, 4)
    ax_03.hist(hazard_rate)
    ax_03.set_xlabel("Hazard Rate")
    sns.despine()

    # Correlations
    # ------------

    ax_10 = plt.subplot(4, 4, 5)
    plot_correlation(
        sum_scores_q1["response"], thetas, r"Sum Score", r"Latent Factor", ax=ax_10
    )

    ax_11 = plt.subplot(4, 4, 6)
    plot_correlation(hazard_rate, thetas, r"Hazard Rate", r"Latent Factor", ax=ax_11)

    ax_12 = plt.subplot(4, 4, 7)
    plot_correlation(
        np.rad2deg(sim_est_errs["main"]),
        thetas,
        r"Estimation Error",
        r"Latent Factor",
        ax=ax_12,
    )

    ax_13 = plt.subplot(4, 4, 8)
    plot_correlation(
        np.rad2deg(sim_est_errs["main"]),
        hazard_rate,
        r"Estimation Error",
        r"Hazard Rate",
        ax=ax_13,
    )

    ax_20 = plt.subplot(4, 4, 9)
    plot_correlation(
        np.rad2deg(sim_est_errs["main"]),
        sum_scores_q1["response"],
        r"Estimation Error",
        r"Sum Scores",
        ax=ax_20,
    )

    ax_21 = plt.subplot(4, 4, 10)
    plot_correlation(
        df_reg["beta_1"], thetas, r"Fixed Learning Rate", r"Latent Factor", ax=ax_21
    )

    ax_22 = plt.subplot(4, 4, 11)
    plot_correlation(
        df_reg["beta_4"],
        thetas,
        r"Adaptive Learning Rate",
        r"Latent Factor",
        ax=ax_22,
    )

    ax_23 = plt.subplot(4, 4, 12)
    plot_correlation(
        df_reg["beta_1"],
        sum_scores_q1["response"],
        r"Fixed Learning Rate",
        r"Sum Scores",
        ax=ax_23,
    )

    ax_30 = plt.subplot(4, 4, 13)
    plot_correlation(
        df_reg["beta_4"],
        sum_scores_q1["response"],
        r"Adaptive Learning Rate",
        r"Sum Scores",
        ax=ax_30,
    )

    ax_31 = plt.subplot(4, 4, 14)
    plot_correlation(
        df_reg["beta_1"],
        hazard_rate,
        r"Fixed Learning Rate",
        r"Hazard Rate",
        ax=ax_31,
    )

    ax_32 = plt.subplot(4, 4, 15)
    plot_correlation(
        df_reg["beta_4"],
        hazard_rate,
        r"Adaptive Learning Rate",
        r"Hazard Rate",
        ax=ax_32,
    )

    # Save figure
    plt.tight_layout()
    plt.savefig("figures/fa_simulation", dpi=400)

    # Check that we have low correlations with eta and sum scores
    # -----------------------------------------------------------

    plt.figure(figsize=(cm2inch(15, 10)))

    ax_11 = plt.subplot(2, 3, 1)
    plot_correlation(hazard_rate, etas, r"Hazard Rate", r"Latent Factor", ax=ax_11)

    ax_12 = plt.subplot(2, 3, 2)
    plot_correlation(
        df_reg["beta_1"], etas, r"Fixed Learning Rate", r"Latent Factor", ax=ax_12
    )

    ax_13 = plt.subplot(2, 3, 3)
    plot_correlation(
        df_reg["beta_4"], etas, r"Adaptive Learning Rate", r"Latent Factor", ax=ax_13
    )

    ax_21 = plt.subplot(2, 3, 4)
    plot_correlation(
        hazard_rate, sum_scores_q2["response"], r"Hazard Rate", r"Sum Scores", ax=ax_21
    )

    ax_22 = plt.subplot(2, 3, 5)
    plot_correlation(
        df_reg["beta_1"],
        sum_scores_q2["response"],
        r"Fixed Learning Rate",
        r"Sum Scores",
        ax=ax_22,
    )

    ax_23 = plt.subplot(2, 3, 6)
    plot_correlation(
        df_reg["beta_4"],
        sum_scores_q2["response"],
        r"Adaptive Learning Rate",
        r"Sum Scores",
        ax=ax_23,
    )

    plt.tight_layout()

    # Plot most important correlations for talk slides
    # ------------------------------------------------

    plt.figure(figsize=(cm2inch(21, 8)))
    ax_00 = plt.subplot(1, 4, 1)
    plot_correlation(
        sum_scores_q1["response"], thetas, r"Sum Score", r"Latent Factor", ax=ax_00
    )

    ax_01 = plt.subplot(1, 4, 2)
    plot_correlation(hazard_rate, thetas, r"Hazard Rate", r"Latent Factor", ax=ax_01)

    ax_02 = plt.subplot(1, 4, 3)
    plot_correlation(
        df_reg["beta_1"], thetas, r"Fixed Learning Rate", r"Latent Factor", ax=ax_02
    )

    ax_03 = plt.subplot(1, 4, 4)
    plot_correlation(
        df_reg["beta_1"],
        sum_scores_q1["response"],
        r"Fixed Learning Rate",
        r"Sum Scores",
        ax=ax_03,
    )

    # Save figure
    plt.tight_layout()
    plt.savefig("figures/sumscore_simulation_talk", dpi=400)

    plt.ioff()
    plt.show()
