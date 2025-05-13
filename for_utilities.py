import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from pycircstat2.utils import angular_distance

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from all_in import cm2inch


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


def plot_robust_regression(
    df_questionnaires: pd.DataFrame,
    df_dependent: pd.DataFrame,
    x_var_name: str,
    xlabel: str,
    ylabel: str,
    use_corr: bool = False,
) -> None:
    """This function plots questionnaire correlations with learning parameters.

    Parameters
    ----------
    df_questionnaires : pd.DataFrame
        Questionnaire data frame.
    df_dependent : pd.DataFrame
        Dependent variable.
    x_var_name : str
        Name of independent variable.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    use_corr : bool
        Optional additional correlation line for checking.

    Returns
    -------
    None
        This function does not return any value.
    """

    # Drop gender = 3 due to low number
    drop_gender = df_questionnaires["gender"] != 3
    df_questionnaires = df_questionnaires[drop_gender].reset_index(drop=True)
    df_dependent = df_dependent[drop_gender].reset_index(drop=True)

    # Instance of standard-scaler class for z-scoring each variable
    scaler_x = StandardScaler()  # for questionnaire data
    scaler_y = StandardScaler()  # for dependent (e.g., fixed LR)
    scaler_z = StandardScaler()  # age

    # Create data frame for regression with z-scored continuous variables
    exog = pd.DataFrame(index=df_questionnaires.index)
    exog["var_x_z"] = scaler_x.fit_transform(df_questionnaires[[x_var_name]])
    exog["age_z"] = scaler_z.fit_transform(df_questionnaires[["age"]])
    exog["gender"] = df_questionnaires[["gender"]].copy()
    exog = sm.add_constant(exog)

    # Z-scored dependent variable
    y_std = scaler_y.fit_transform(df_dependent)

    # Run regression
    rlm_model = sm.RLM(y_std, exog, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()

    # Extract coefficient of questionnaire variable and associated p-value
    r = round(rlm_results.params["var_x_z"], 2)
    p = round(rlm_results.pvalues["var_x_z"], 2)

    # Scatter plot of single subjects
    plt.scatter(df_questionnaires[[x_var_name]], df_dependent, alpha=0.6)

    # Fit a line
    if use_corr:
        slope, intercept = np.polyfit(df_questionnaires[x_var_name], df_dependent, 1)
        plt.plot(
            df_questionnaires[x_var_name],
            slope * df_questionnaires[x_var_name] + intercept,
            color="red",
        )

        # Compute correlation and put in title
        r, p = stats.pearsonr(df_questionnaires[x_var_name], df_dependent.iloc[:, 0])

    # Create data frame with values in [min, max] of dependent variable for model predictions
    x_grid = pd.DataFrame(
        np.linspace(
            df_questionnaires[x_var_name].min(),
            df_questionnaires[x_var_name].max(),
            100,
        ),
        columns=[x_var_name],  # same column name used for fit_transform
    )

    # Standardize x grid using scaler from above
    x_grid_std = scaler_x.transform(x_grid)

    # Create prediction input with gender fixed to 0
    gender_fixed = 0  # average of -1 and 1
    age_fixed_z = 0  # standardized mean

    # Build data frame for prediction plot
    exog_plot = pd.DataFrame(
        {
            "const": 1,
            "var_x_z": x_grid_std.flatten(),
            "age_z": age_fixed_z,
            "gender": gender_fixed,
        }
    )

    # Predict based on fixed variables
    y_pred_std_grid = rlm_results.predict(exog_plot)
    y_pred_grid = scaler_y.inverse_transform(
        np.array(y_pred_std_grid).reshape(-1, 1)
    ).flatten()

    # Plot predictions
    plt.plot(x_grid.values.flatten(), y_pred_grid, color="blue")

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"r = {r:.3f}, p = {p:.3f}")

    # Delete unnecessary axes
    sns.despine()


def plot_main_questionnaire_correlations(
    df_questionnaires: pd.DataFrame,
    df_dependent: pd.DataFrame,
    dep_var_name: str,
    ylabel: str,
    fig_width: int = 10,
    fig_height: int = 10,
    use_corr: bool = False,
) -> None:
    """This function plots questionnaire correlations with learning parameters.

    Parameters
    ----------
    df_questionnaires : pd.DataFrame
        Questionnaire data frame.
    df_dependent : pd.DataFrame
        Dependent variable.
    dep_var_name : str
        Name of dependent variable.
    ylabel : str
        Y-axis label.
    fig_width : int
        Width of figure in cm.
    fig_height : int
        Height of figure in cm.
    use_corr : bool
        Optional additional correlation line for checking.

    Returns
    -------
    None
        This function does not return any value.

    """
    # Create figure
    plt.figure(figsize=cm2inch(fig_width, fig_height))

    # CAPE sum score
    plt.subplot(321)
    plot_robust_regression(
        df_questionnaires,
        df_dependent[[dep_var_name]],
        "CAPE1",
        "CAPE Score",
        ylabel,
        use_corr=use_corr,
    )

    # CAPE positive symptoms
    plt.subplot(322)
    plot_robust_regression(
        df_questionnaires,
        df_dependent[[dep_var_name]],
        "CAPE_pos",
        "CAPE pos Score",
        ylabel,
        use_corr=use_corr,
    )

    # CAPE negative symptoms
    plt.subplot(323)
    plot_robust_regression(
        df_questionnaires,
        df_dependent[[dep_var_name]],
        "CAPE_neg",
        "CAPE neg Score",
        ylabel,
        use_corr=use_corr,
    )

    # CAPE depressive symptoms
    plt.subplot(324)
    plot_robust_regression(
        df_questionnaires,
        df_dependent[[dep_var_name]],
        "CAPE_dep",
        "CAPE dep Score",
        ylabel,
        use_corr=use_corr,
    )

    # IUS sum score
    plt.subplot(325)
    plot_robust_regression(
        df_questionnaires,
        df_dependent[[dep_var_name]],
        "IUS1",
        "IUS Score",
        ylabel,
        use_corr=use_corr,
    )

    # SPQ sum score
    plt.subplot(326)
    plot_robust_regression(
        df_questionnaires,
        df_dependent[[dep_var_name]],
        "SPQ1",
        "SPQ1 Score",
        ylabel,
        use_corr=use_corr,
    )

    # Delete unnecessary axes
    plt.tight_layout()


def plot_idas_correlations(
    df_questionnaires: pd.DataFrame,
    df_dependent: pd.DataFrame,
    dep_var_name: str,
    ylabel: str,
    fig_width: int = 15,
    fig_height: int = 20,
    use_corr: bool = False,
) -> None:
    """This function plots IDAS subscale correlations with learning parameters.

    Parameters
    ----------
    df_questionnaires : pd.DataFrame
        Questionnaire data frame.
    df_dependent : pd.DataFrame
        Dependent variable.
    dep_var_name : str
        Name of dependent variable.
    ylabel : str
        Y-axis label.
    fig_width : int
        Figure width in cm.
    fig_height : int
        Figure height in cm.
    use_corr : bool
        Optional additional correlation line for checking.

    Returns
    -------
    None
        This function does not return any value.
    """

    # Create figure
    plt.figure(figsize=cm2inch(fig_width, fig_height))

    # Dependent variable name
    y = df_dependent[[dep_var_name]]

    # Dysphoria
    plt.subplot(541)
    plot_robust_regression(
        df_questionnaires, y, "dysphoria", "Dysphoria Score", ylabel, use_corr=use_corr
    )

    # Fatigue
    plt.subplot(542)
    plot_robust_regression(
        df_questionnaires, y, "fatigue", "Fatigue Score", ylabel, use_corr=use_corr
    )

    # Insomnia
    plt.subplot(543)
    plot_robust_regression(
        df_questionnaires, y, "insomnia", "Insomnia Score", ylabel, use_corr=use_corr
    )

    # Suicidality
    plt.subplot(544)
    plot_robust_regression(
        df_questionnaires,
        y,
        "suicidality",
        "Suicidality Score",
        ylabel,
        use_corr=use_corr,
    )

    # Increase in appetite
    plt.subplot(545)
    plot_robust_regression(
        df_questionnaires,
        y,
        "incr_appetite",
        "Increase Appetite Score",
        ylabel,
        use_corr=use_corr,
    )

    # Loss of appetite
    plt.subplot(546)
    plot_robust_regression(
        df_questionnaires,
        y,
        "loss_appetite",
        "Loss Appetite Score",
        ylabel,
        use_corr=use_corr,
    )

    # Wellbeing
    plt.subplot(547)
    plot_robust_regression(
        df_questionnaires, y, "wellbeing", "Wellbeing Score", ylabel, use_corr=use_corr
    )

    # Moodiness
    plt.subplot(548)
    plot_robust_regression(
        df_questionnaires, y, "moodiness", "Moodiness Score", ylabel, use_corr=use_corr
    )

    # Mania
    plt.subplot(549)
    plot_robust_regression(
        df_questionnaires, y, "mania", "Mania Score", ylabel, use_corr=use_corr
    )

    # Euphoria
    plt.subplot(5, 4, 10)
    plot_robust_regression(
        df_questionnaires, y, "euphoria", "Euphoria Score", ylabel, use_corr=use_corr
    )

    # Social anxiety
    plt.subplot(5, 4, 11)
    plot_robust_regression(
        df_questionnaires,
        y,
        "social_anx",
        "Social Anxiety Score",
        ylabel,
        use_corr=use_corr,
    )

    # Claustrophobia
    plt.subplot(5, 4, 12)
    plot_robust_regression(
        df_questionnaires,
        y,
        "claustrophobia",
        "Claustrophobia Score",
        ylabel,
        use_corr=use_corr,
    )

    # Traumatic intrusions
    plt.subplot(5, 4, 13)
    plot_robust_regression(
        df_questionnaires,
        y,
        "traumatic_intrusions",
        "Traumatic Intrusions Score",
        ylabel,
        use_corr=use_corr,
    )

    # Traumatic avoidance
    plt.subplot(5, 4, 14)
    plot_robust_regression(
        df_questionnaires,
        y,
        "traumatic_avoidance",
        "Traumatic Avoidance Score",
        ylabel,
        use_corr=use_corr,
    )

    # Compulsion order
    plt.subplot(5, 4, 15)
    plot_robust_regression(
        df_questionnaires,
        y,
        "compulsion_order",
        "Compulsion Order Score",
        ylabel,
        use_corr=use_corr,
    )

    # Compulsion clean
    plt.subplot(5, 4, 16)
    plot_robust_regression(
        df_questionnaires,
        y,
        "compulsion_clean",
        "Compulsion Clean Score",
        ylabel,
        use_corr=use_corr,
    )

    # Compulsion control
    plt.subplot(5, 4, 17)
    plot_robust_regression(
        df_questionnaires,
        y,
        "compulsion_control",
        "Compulsion Control Score",
        ylabel,
        use_corr=use_corr,
    )

    # Panic
    plt.subplot(5, 4, 18)
    plot_robust_regression(
        df_questionnaires, y, "panic", "Panic Score", ylabel, use_corr=use_corr
    )

    # Delete unnecessary axes
    plt.tight_layout()


def plot_questionnaire_correlations_noise(
    df_questionnaires: pd.DataFrame,
    df_regression_low_noise: pd.DataFrame,
    df_regression_high_noise: pd.DataFrame,
    dep_var_name: str,
    ylabel: str,
    fig_width: int = 10,
    fig_height: int = 10,
    use_corr: bool = False,
) -> None:
    """This function plot correlations separately for the low- and high-noise conditions.

    Parameters
    ----------
    df_questionnaires : pd.DataFrame
        Questionnaire dataframe.
    df_regression_low_noise : pd.DataFrame
        Regression data frame low noise.
    df_regression_high_noise : pd.DataFrame
        Regression data frame high noise.
    dep_var_name : str
        Dependent variable name.
    ylabel : str
        Y-axis label.
    fig_width : int
        Figure width in cm.
    fig_height : int
        Figure height in cm.
    use_corr : bool
        Optional additional correlation line for checking.

    Returns
    -------
    None
        This function does not return any value.
    """

    # Create figure
    plt.figure(figsize=cm2inch(fig_width, fig_height))

    # CAPE low noise
    plt.subplot(321)
    plot_robust_regression(
        df_questionnaires,
        df_regression_low_noise[[dep_var_name]],
        "CAPE1",
        "CAPE1 Score Low Noise",
        ylabel,
        use_corr=use_corr,
    )

    # CAPE high noise
    plt.subplot(322)
    plot_robust_regression(
        df_questionnaires,
        df_regression_high_noise[[dep_var_name]],
        "CAPE1",
        "CAPE1 Score High Noise",
        ylabel,
        use_corr=use_corr,
    )

    # IUS low noise
    plt.subplot(323)
    plot_robust_regression(
        df_questionnaires,
        df_regression_low_noise[[dep_var_name]],
        "IUS1",
        "IUS Score Low Noise",
        ylabel,
        use_corr=use_corr,
    )

    # IUS high noise
    plt.subplot(324)
    plot_robust_regression(
        df_questionnaires,
        df_regression_high_noise[[dep_var_name]],
        "IUS1",
        "IUS Score High Noise",
        ylabel,
        use_corr=use_corr,
    )

    # SPQ low noise
    plt.subplot(325)
    plot_robust_regression(
        df_questionnaires,
        df_regression_low_noise[[dep_var_name]],
        "SPQ1",
        "SPQ1 Score Low Noise",
        ylabel,
        use_corr=use_corr,
    )

    # SPQ high noise
    plt.subplot(326)
    plot_robust_regression(
        df_questionnaires,
        df_regression_high_noise[[dep_var_name]],
        "SPQ1",
        "SPQ1 Score High Noise",
        ylabel,
        use_corr=use_corr,
    )

    # Delete unnecessary axes
    plt.tight_layout()
