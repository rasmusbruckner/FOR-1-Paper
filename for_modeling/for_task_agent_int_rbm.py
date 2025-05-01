"""Task-Agent Interaction: Interaction between reduced Bayesian model and predictive inference task."""

from typing import Tuple

import numpy as np
import pandas as pd
from rbm_analyses import AgentVars, AlAgent, residual_fun
from rbm_analyses.utilities import circ_dist
from scipy.stats import vonmises


def task_agent_int(
    df: pd.DataFrame,
    agent: AlAgent,
    agent_vars: AgentVars,
    sel_coeffs: dict,
    sim: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """This function models the interaction between task and agent (RBM).

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with relevant data.
    agent : "AlAgent"
        Agent-object instance.
    agent_vars : AgentVars
        Agent-variables-object instance.
    sel_coeffs : dict
        Selected model parameters.
    sim : bool
        Indicates whether predictions are simulated or models variables merely extracted.

    Returns
    -------
    np.ndarray
        Log-likelihood values.
    pd.DataFrame
        Simulation results.
    """

    # Extract and initialize relevant variables
    # -----------------------------------------
    n_trials = len(df)  # number of trials
    mu = np.full(
        [n_trials], np.nan
    )  # inferred mean of the outcome-generating distribution
    a_hat = np.full(
        n_trials, np.nan
    )  # predicted update according to reduced Bayesian model
    epsilon = np.full(n_trials, np.nan)  # response noise
    omega = np.full(n_trials, np.nan)  # change-point probability
    tau = np.full(n_trials, np.nan)  # relative uncertainty
    alpha = np.full(n_trials, np.nan)  # learning rate
    sigma_t_sq = np.full(n_trials, np.nan)  # estimation uncertainty

    # Prediction error
    if not sim:
        delta = df["delta_t_rad"].copy()
    else:
        delta = np.full(len(df), np.nan)

    # Log-likelihood
    n_new_block = np.sum(df["new_block"] == 1)
    llh_rbm = np.full(
        [n_trials - n_new_block], np.nan
    )  # log-likelihood of reduced Bayesian model

    llh_sim = np.full([n_trials], np.nan)  # log-likelihood of simulated update

    # Initialize variables related to simulations
    sim_b_t = np.full(n_trials, np.nan)  # simulated prediction
    sim_a_t = np.full(n_trials, np.nan)  # simulated update

    # Initialize variables related to estimation
    llh_counter = 0
    corrected_0_p = 1e-10

    # Cycle over trials
    # -----------------
    for t in range(0, n_trials - 1):

        # Extract noise condition
        agent.sigma = df["sigma"][t].copy()

        # For first trial of new block
        # Futuretodo: create function to re-initialize agent on new block, maybe shared across motor and sampling too
        if df["new_block"][t]:

            # Initialize estimation uncertainty, relative uncertainty, and changepoint probability
            agent.sigma_t_sq = agent_vars.sigma_0
            agent.tau_t = agent_vars.tau_0
            agent.omega_t = agent_vars.omega_0

            # Record estimation uncertainty
            sigma_t_sq[t] = agent_vars.sigma_0

            if sim:
                # Set initial prediction
                sim_b_t[t] = agent_vars.mu_0

        # Record relative uncertainty of current trial
        tau[t] = agent.tau_t

        # Record estimation uncertainty of current trial
        sigma_t_sq[t] = agent.sigma_t_sq

        # For all but last trials of a block:
        if not df["new_block"][t + 1]:

            # No reward manipulation here
            high_val = 0

            # Sequential belief update
            if sim:
                delta[t] = circ_dist(df["x_t_rad"][t], sim_b_t[t])

                agent.learn(
                    delta[t], sim_b_t[t], df["v_t"][t], df["mu_t_rad"][t], high_val
                )
            else:
                agent.learn(
                    delta[t],
                    df["b_t_rad"][t],
                    df["v_t"][t],
                    df["mu_t_rad"][t],
                    high_val,
                )

            # Record updated belief
            mu[t] = agent.mu_t

            # Record predicted update according to reduced Bayesian model
            a_hat[t] = agent.a_t

            # Record change-point probability
            omega[t] = agent.omega_t

            # Record learning rate
            alpha[t] = agent.alpha_t

            # Compute likelihood of updates according to reduced Bayesian model
            # -----------------------------------------------------------------

            # Compute absolute predicted update
            # |hat{a}_t|
            abs_pred_up = abs(a_hat[t])

            # Compute response noise
            epsilon[t] = residual_fun(abs_pred_up, sel_coeffs['omikron_0'], sel_coeffs['omikron_1'])

            # Compute likelihood of predicted update
            # p(a_t) := N(a_t; hat{a}_t, epsilon_t^2)
            p_a_t = vonmises.pdf(df["a_t_rad"][t], loc=a_hat[t], kappa=epsilon[t])

            # Adjust probability of update for numerical stability
            if p_a_t == 0.0:
                p_a_t = corrected_0_p

            # Compute negative log-likelihood of predicted update according to reduced Bayesian model
            llh_rbm[llh_counter] = np.log(p_a_t)

            # Simulate updates
            if sim:

                # Sample updates from von Mises distribution
                sim_a_t[t] = np.random.vonmises(a_hat[t], epsilon[t])
                p_a_t_sim = vonmises.pdf(sim_a_t[t], loc=a_hat[t], kappa=epsilon[t])

                # Adjust probability of update for numerical stability
                if p_a_t_sim == 0.0:
                    p_a_t_sim = corrected_0_p

                # Compute negative log-likelihood of predicted update according to reduced Bayesian model
                llh_sim[t] = np.log(p_a_t_sim)

                # Updated prediction
                sim_b_t[t + 1] = (sim_b_t[t] + sim_a_t[t]) % agent.max_x

            llh_counter += 1

    # Attach model variables to data frame
    df_data = pd.DataFrame(index=range(0, n_trials), dtype="float")
    df_data["a_t_rad_hat"] = a_hat
    df_data["mu_t_rad"] = mu
    df_data["omega_t"] = omega
    df_data["tau_t"] = tau
    df_data["alpha_t"] = alpha
    df_data["sigma_t_sq"] = sigma_t_sq

    if sim:
        # Save simulation-related variables
        df_data["sim_b_t_rad"] = sim_b_t
        df_data["sim_a_t_rad"] = sim_a_t
        df_data["sim_a_t_llh"] = llh_sim
        df_data["delta_t_rad"] = delta
        df_data["sigma"] = df["sigma"].copy()
        df_data["new_block"] = df["new_block"].copy()
        df_data["x_t_rad"] = df["x_t_rad"].copy()
        df_data["v_t"] = df["v_t"].copy()

    return llh_rbm, df_data
