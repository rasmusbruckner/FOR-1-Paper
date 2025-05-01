"""Simulations reduced Bayesian model: Run simulations across whole data set, e.g.,
for posterior predictive checks."""

from time import sleep
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from all_in import get_df_subj
from rbm_analyses import AgentVars, AlAgent
from tqdm import tqdm

from FOR_1_Paper.for_modeling.for_task_agent_int_rbm import task_agent_int
from FOR_1_Paper.for_utilities import get_sim_est_err


def simulation(
    df_exp: pd.DataFrame,
    df_model: pd.DataFrame,
    n_subj: int,
    plot_data: bool = False,
    sim: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """This function simulates data using the RBM mixture model.

    Parameters
    ----------
    df_exp : pd.DataFrame
        Data frame containing participant data.
    df_model : pd.DataFrame
        Data frame containing model parameters.
    n_subj : int
        Number of participants.
    plot_data : bool
        Indicates if single-trial plots for updates and predictions should be generated.
    sim : bool
        Indicates if prediction errors are simulated or not.

    Returns
    -------
    pd.DataFrame
        sim_est_err with simulated estimation errors.
    pd.DataFrame
         df_sim with simulation results.
    pd.DataFrame
        true_params with true parameters.
    """

    # Inform user
    sleep(0.1)
    print("\nModel simulation:")
    sleep(0.1)

    # Initialize progress bar
    pbar = tqdm(total=n_subj)

    # Agent-variables object
    agent_vars = AgentVars()
    agent_vars.max_x = 2 * np.pi
    agent_vars.mu_0 = 0

    # Initialize data frame for data that will be recovered
    df_sim = pd.DataFrame()

    # Initialize data frame for estimation errors
    sim_est_err = pd.DataFrame(columns=["main"], index=np.arange(n_subj), dtype=float)

    # Initialize true param
    true_params = np.nan

    # Cycle over participants
    # -----------------------
    for i in range(0, n_subj):

        # Extract subject-specific data frame
        df_subj = get_df_subj(df_exp, i)

        # Extract model parameters from model data frame
        sel_coeffs = df_model[df_model["subj_num"] == i + 1].copy()

        # Save parameters for parameter-recovery analysis
        if i == 0:
            true_params = sel_coeffs
        else:
            true_params = pd.concat([true_params, sel_coeffs], ignore_index=True)

        # Select relevant variables from parameter data frame
        sel_coeffs = sel_coeffs[
            ["omikron_0", "omikron_1", "h", "s", "u", "sigma_H"]
        ].iloc[0].to_dict()

        # Set agent variables of current participant
        agent_vars.h = sel_coeffs['h']
        agent_vars.s = sel_coeffs['s']
        agent_vars.u = np.exp(sel_coeffs['u'])
        agent_vars.sigma_H = sel_coeffs['sigma_H']

        # For consistency with Matt
        agent_vars.tau_0 = 0.99
        agent_vars.sigma_0 = (
            6.1875  # due to his initRU settings that I recomputed to sigma_0
        )

        # Ensure we use the circular model version
        agent_vars.circular = True

        # Agent object
        agent = AlAgent(agent_vars)

        # Run task-agent interaction
        _, df_data = task_agent_int(df_subj, agent, agent_vars, sel_coeffs, sim=sim)

        # Record subject number
        df_data["subj_num"] = i + 1

        # Add data to data frame
        df_sim = pd.concat([df_sim, df_data], ignore_index=True)

        # Plot for qualitative checks
        if plot_data:
            # Plot updates
            plt.figure()
            plt.plot(np.arange(len(df_subj["a_t_rad"])), df_subj["a_t_rad"])
            plt.plot(np.arange(len(df_data["a_t_rad_hat"])), df_data["a_t_rad_hat"])
            plt.legend(["a_t_rad", "a_t_rad_hat"], loc=1, framealpha=0.8)

            # Save the plot
            savename = "figures/single_trial/up_%s.pdf" % i
            plt.savefig(savename)
            plt.close()

            # Plot predictions
            plt.figure()

            # Correct for circular issues
            wrapped_outcomes = np.mod(df_subj["x_t_rad"] + np.pi, 2 * np.pi)
            wrapped_pred = np.mod(df_subj["b_t_rad"] + np.pi, 2 * np.pi)
            wrapped_sim_pred = np.mod(df_data["sim_b_t_rad"] + np.pi, 2 * np.pi)

            plt.plot(np.arange(len(df_subj["x_t_rad"])), wrapped_outcomes, ".")
            plt.plot(np.arange(len(df_subj["b_t_rad"])), wrapped_pred, ".")
            plt.plot(np.arange(len(df_data["sim_b_t_rad"])), wrapped_sim_pred, ".")
            plt.legend(["x_t_rad", "b_t_rad", "sim_b_t_rad"], loc=1, framealpha=0.8)

            # Save the plot
            savename = "figures/single_trial/bel_%s.pdf" % i
            plt.savefig(savename)
            plt.close()

        # Extract estimation error
        if sim:
            sim_est_err_main = get_sim_est_err(df_subj, df_data)
            sim_est_err.loc[i, "main"] = sim_est_err_main

        # Update progress bar
        pbar.update(1)

        # Close progress bar
        if i == n_subj - 1:
            pbar.close()

    return sim_est_err, df_sim, true_params


def simulation_loop(
    df_exp: pd.DataFrame,
    df_model: pd.DataFrame,
    n_subj: int,
    plot_data: bool = False,
    sim: bool = False,
    n_sim: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """This function runs the simulation across multiple cycles.

    Parameters
    ----------
    df_exp : pd.DataFrame
        Data frame containing participant data.
    df_model : pd.DataFrame
        Data frame containing model parameters.
    n_subj : int
        Number of participants.
    plot_data : bool
        Indicates if single-trial plots for updates and predictions should be generated.
    sim : bool
        Indicates if prediction errors are simulated or not.
    n_sim : int
        Number of simulations.

    Returns
    -------
    pd.DataFrame
        Simulated estimation errors of all cycles.
    pd.DataFrame
        Simulation results.
    """

    # Initiale data frames
    all_sim_est_errs = np.nan
    all_data = np.nan

    # Cycle over simulations
    for i in range(0, n_sim):

        # Simulate the data
        sim_est_err, df_sim, _ = simulation(
            df_exp, df_model, n_subj, plot_data=plot_data, sim=sim
        )

        # Put all data in data frame for estimation errors
        if i == 0:
            all_sim_est_errs = sim_est_err
            all_data = df_sim.copy()
        else:
            all_sim_est_errs = pd.concat([all_sim_est_errs, sim_est_err])
            all_data = pd.concat([all_data, df_sim])

    return all_sim_est_errs, all_data
