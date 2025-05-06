import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from pycircstat2.utils import angular_distance


def safe_save_dataframe(dataframe: pd.DataFrame) -> None:
    """Saves a data frame and ensures that values don't change unexpectedly.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Data frame to be saved.

    Returns
    -------
    None
        This function does not return any value.
    """

    # Initialize expected data frame
    expected_df = np.nan

    # Load previous file for comparison
    # ---------------------------------

    # File path
    df_name = "for_data/" + dataframe.name + ".pkl"

    # Check if file exists
    path_exist = os.path.exists(df_name)

    # If so, load file for comparison
    if path_exist:
        expected_df = pd.read_pickle(df_name)

    # If we have the file already, check if as expected
    if path_exist:
        # Test if equal and save data
        same = dataframe.equals(expected_df)
        print("\nActual and expected " + dataframe.name + " equal:", same, "\n")

    # If new, we'll create the file
    else:
        same = True
        dataframe.to_pickle("for_data/" + dataframe.name + ".pkl")
        print("\nCreating new data frame: " + dataframe.name + "\n")

    if not same:
        dataframe.to_pickle("for_data/" + dataframe.name + "_unexpected.pkl")
        print("\nCreating new data frame: " + dataframe.name + "_unexpected.pkl" + "\n")


def recovery_summary(
    true_params: pd.DataFrame,
    est_params: pd.DataFrame,
    param_names: list[str],
    grid_size: tuple,
) -> None:
    """Creates a simple plot showing recovery results.

    Parameters
    ----------
    true_params : pd.DataFrame
        Ground truth parameters.
    est_params : pd.DataFrame
        Estimated parameter values.
    param_names : list[str]
        List of strings with parameter names.
    grid_size : tuple
        Indicates plot grid size (rows, columns).

    Returns
    -------
    None
        This function does not return any value.
    """

    # Create figure
    plt.figure()

    # Cycle over parameters
    for i, label in enumerate(param_names):

        # Create subplot
        plt.subplot(grid_size[0], grid_size[1], i + 1)
        plt.title(f"{label}")

        # Extract parameter values
        true_param_value = true_params[label]
        est_param_value = est_params[label]

        # Plot parameters
        plt.scatter(true_param_value, est_param_value, marker="o")

        # Compute Spearman correlation
        r, _ = stats.spearmanr(true_param_value, est_param_value)
        plt.title(f"{label}: r={round(r, 2)}")

        # Add axis labels
        plt.xlabel("True parameter")
        plt.ylabel("Estimated parameter")

    # Adjust layout and show plot
    plt.tight_layout()


def get_sim_est_err(df_subj: pd.DataFrame, df_data: pd.DataFrame) -> float:
    """This function computes the simulated estimation errors.

    Parameters
    ----------
    df_subj : pd.DataFrame
        Subject data with ground truth mu.
    df_data : pd.DataFrame
        Simulated data with mu estimates.

    Returns
    -------
    float
        Mean simulated estimation error w/o change-point trials.
    """

    # Extract no-changepoint trials
    no_cp = df_subj["c_t"].to_numpy() == 0

    # Extract true helicopter location for estimation-error computation
    real_mu = df_subj["mu_t_rad"].to_numpy()

    # Extract model prediction for estimation-error computation
    sim_pred = df_data["sim_b_t_rad"].to_numpy()

    # Compute estimation error
    sim_est_err_all = angular_distance(real_mu, sim_pred)
    sim_est_err_nocp = sim_est_err_all[no_cp]  # estimation error without change points
    sim_est_err = np.mean(abs(sim_est_err_nocp))

    return sim_est_err


def plot_questionnaire_correlation(
    x: pd.Series, y: pd.Series, xlabel: str, ylabel: str
) -> None:
    """This function plots questionnaire correlations with learning parameters.

    Parameters
    ----------
    x : pd.Series
        Regression results.
    y : pd.Series
        Questionnaire scores.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.

    Returns
    -------
    None
        This function does not return any value.
    """

    # Todo: optionally spearman correlation

    # Scatter plot of single subjects
    plt.scatter(x, y, alpha=0.6, label="Predicted mean")

    # Fit a line
    slope, intercept = np.polyfit(x, y, 1)
    plt.plot(x, slope * x + intercept, color="blue")

    # Add labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Compute correlation and put in title
    r, p = stats.pearsonr(x, y)
    plt.title(f"r = {r:.3f}, p = {p:.3f}")

    # Delete unnecessary axes
    sns.despine()
