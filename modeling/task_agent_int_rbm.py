"""Task-Agent Interaction: Interaction between reduced Bayesian model and predictive inference task."""

from typing import Tuple

import numpy as np
import pandas as pd
from rbmpy import AgentVars, AlAgent, residual_fun
from rbmpy.task.ChangePointTask import ChangePointTask
from rbmpy.task.TaskVars import TaskVars
from rbmpy.utilities import circ_dist, compute_persprob
from scipy.stats import vonmises

from FOR_1_Paper.for_utilities import get_sim_est_err


def task_agent_int(
    df_subj: pd.DataFrame,
    agent: AlAgent,
    agent_vars: AgentVars,
    sel_coeffs: dict,
    sim: str | bool = False,
    mixture: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """This function models the interaction between task and agent (RBM).

    Parameters
    ----------
    df_subj : pd.DataFrame
        Data frame with relevant data.
    agent : AlAgent
        Agent-object instance.
    agent_vars : AgentVars
        Agent-variables-object instance.
    sel_coeffs : dict
        Selected model parameters.
    sim : str | bool
        False = no simulation, only likelihood evaluation for empirical task data (default).
        agent = simulate agent predictions based on empirical task data.
        task_agent = simulate agent predictions based on simulated task data.
    mixture : bool
        Indicates whether to use mixture model or not (default).

    Returns
    -------
    np.ndarray
        Log-likelihood values.
    pd.DataFrame
        Simulated agent behavior.
    """

    # Extract and initialize relevant variables
    # -----------------------------------------
    n_trials = len(df_subj)  # number of trials
    mu = np.full(
        [n_trials], np.nan
    )  # inferred mean of the outcome-generating distribution
    a_hat = np.full(
        n_trials, np.nan
    )  # predicted update according to reduced Bayesian model
    concentration = np.full(n_trials, np.nan)  # response noise
    omega = np.full(n_trials, np.nan)  # change-point probability
    tau = np.full(n_trials, np.nan)  # relative uncertainty
    alpha = np.full(n_trials, np.nan)  # learning rate
    sigma_t_sq = np.full(n_trials, np.nan)  # estimation uncertainty
    hit = np.full(len(df_subj), np.nan)

    # Prediction error
    if not sim:
        delta = df_subj["delta_t_rad"].copy()
    else:
        delta = np.full(len(df_subj), np.nan)

    # Log-likelihood
    n_new_block = np.sum(df_subj["new_block"] == 1)
    llh_rbm = np.full([n_trials - n_new_block], np.nan)  # RBM
    llh_sim = np.full([n_trials], np.nan)  # simulated update

    # Initialize variables related to simulations
    sim_b_t = np.full(n_trials, np.nan)  # simulated prediction
    sim_a_t = np.full(n_trials, np.nan)  # simulated update

    task = None
    if sim == "task_agent":

        # Initialize task
        task_vars = TaskVars()
        task_vars.circular = True
        task = ChangePointTask(task_vars)

        # Initialize task variables
        df_subj["c_t"] = np.nan
        df_subj["mu_t_rad"] = np.nan
        df_subj["x_t_rad"] = np.nan
        df_subj["v_t"] = np.nan

    # Initialize variables related to estimation
    llh_counter = 0
    corrected_0_p = 1e-10

    # Cycle over trials
    # -----------------
    for t in range(0, n_trials - 1):

        # Extract noise condition
        agent.sigma = df_subj["sigma"][t].copy()

        # For first trial of new block
        # Futuretodo: create function to re-initialize agent on new block, maybe shared across motor and sampling too
        if df_subj["new_block"][t]:

            # Reset task variables to ensure change-point on first trial of new block
            if sim == "task_agent":
                task.new_block = 1

            # Initialize estimation uncertainty, relative uncertainty, and change-point probability
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
        if not df_subj["new_block"][t + 1]:

            # No reward manipulation here
            high_val = 0

            # Sequential belief update
            if sim == "agent":

                delta[t] = circ_dist(df_subj["x_t_rad"][t], sim_b_t[t])
                agent.learn(
                    float(delta[t]),
                    float(sim_b_t[t]),
                    df_subj["v_t"][t],
                    df_subj["mu_t_rad"][t],
                    high_val,
                )

            elif sim == "task_agent":

                # Generate task outcomes for current trial
                task.kappa = df_subj["kappa_t"][t].copy()
                task.sample_cp()
                task.sample_mu()
                task.sample_outcome()
                task.sample_catch_trial()
                task.new_block = 0

                # Save task variables
                df_subj.loc[t, "c_t"] = task.cp
                df_subj.loc[t, "x_t_rad"] = task.x_t
                df_subj.loc[t, "v_t"] = task.catch_trial
                df_subj.loc[t, "mu_t_rad"] = task.mu

                # Compute prediction error
                delta[t] = circ_dist(task.x_t, sim_b_t[t])

                # Run agent
                agent.learn(
                    float(delta[t]),
                    float(sim_b_t[t]),
                    df_subj["v_t"][t],
                    task.mu,
                    high_val,
                )

            elif not sim:

                agent.learn(
                    float(delta[t]),
                    df_subj["b_t_rad"][t],
                    df_subj["v_t"][t],
                    df_subj["mu_t_rad"][t],
                    high_val,
                )

            else:
                raise ValueError("sim must be either 'agent' or 'task_agent' or False")

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
            concentration[t] = residual_fun(
                abs_pred_up, sel_coeffs["omikron_0"], sel_coeffs["omikron_1"]
            )

            if not sim:

                # Compute likelihood of predicted update
                # p(a_t) := N(a_t; hat{a}_t, epsilon_t^2) (in terms of Gaussian)
                p_a_t = vonmises.pdf(
                    df_subj["a_t_rad"][t], loc=a_hat[t], kappa=concentration[t]
                )

                # Adjust probability of update for numerical stability
                if p_a_t == 0.0:
                    p_a_t = corrected_0_p

                # Compute log-likelihood of predicted update according to reduced Bayesian model
                llh_rbm[llh_counter] = np.log(p_a_t)

                # Implement hard mixture: We assumed update is either generated by
                # perseveration model or reduced Bayesian model.
                # An alternative would be a soft mixture between both models.
                if mixture:

                    # Compute perseveration probability
                    lambda_t = compute_persprob(
                        sel_coeffs["lambda_0"],
                        sel_coeffs["lambda_1"],
                        np.rad2deg(abs_pred_up),
                    )

                    # Adjust lambda for numerical stability
                    if lambda_t == 0:
                        lambda_t = corrected_0_p
                    if lambda_t == 1:
                        lambda_t = 1 - corrected_0_p

                    # Identify perseveration trials
                    pers = df_subj["a_t_rad"][t] == 0

                    # Compute log-likelihood according to perseveration model
                    if pers:
                        llh_rbm[llh_counter] = np.log(lambda_t)

                    # Compute log-likelihood according to reduced Bayesian model
                    elif not pers:
                        llh_rbm[llh_counter] = (
                            np.log(1 - lambda_t) + llh_rbm[llh_counter]
                        )

            # Simulate updates
            elif sim == "agent" or sim == "task_agent":

                lambda_t = None
                if mixture:

                    # Compute perseveration probability
                    lambda_t = compute_persprob(
                        sel_coeffs["lambda_0"],
                        sel_coeffs["lambda_1"],
                        np.rad2deg(abs_pred_up),
                    )

                    # Randomly sample perseveration trials
                    rand_pers = np.random.binomial(1, lambda_t)
                else:
                    rand_pers = 0

                # Sample update
                if rand_pers == 0:
                    # Sample update from von Mises distribution
                    sim_a_t[t] = np.random.vonmises(a_hat[t], concentration[t])
                    p_a_t_sim = vonmises.pdf(sim_a_t[t], loc=a_hat[t], kappa=concentration[t])
                else:
                    # Perseveration
                    sim_a_t[t] = 0.0
                    p_a_t_sim = lambda_t

                # Adjust probability of update for numerical stability
                if p_a_t_sim == 0.0:
                    p_a_t_sim = corrected_0_p

                # Compute log-likelihood of predicted update
                llh_sim[t] = np.log(p_a_t_sim)

                # Updated prediction
                sim_b_t[t + 1] = (sim_b_t[t] + sim_a_t[t]) % agent.max_x

                # Record hit vs. miss
                if abs(delta[t]) <= df_subj["angular_shield_size"][t] / 2:
                    hit[t] = 1
                else:
                    hit[t] = 0

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
        df_data["sigma"] = df_subj["sigma"].copy()
        df_data["kappa"] = df_subj["kappa_t"].copy()
        df_data["group"] = df_subj["group"].copy()
        df_data["new_block"] = df_subj["new_block"].copy()
        df_data["x_t_rad"] = df_subj["x_t_rad"].copy()
        df_data["v_t"] = df_subj["v_t"].copy()
        df_data["c_t"] = df_subj["c_t"].copy()
        df_data["task_mu"] = df_subj["mu_t_rad"].copy()
        df_data["hit"] = hit

        # Compute estimation error
        _, sim_est_err_all = get_sim_est_err(df_subj, df_data)
        df_data["sim_e_t_rad"] = sim_est_err_all

    return llh_rbm, df_data
