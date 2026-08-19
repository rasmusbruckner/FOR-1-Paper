import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from FOR_1_Paper.modeling.simulation_rbm import simulation_loop
from FOR_1_Paper.regression.ForRegVars import RegVars
from FOR_1_Paper.regression.RegressionFor import RegressionFor


def normal_ogive(b: np.ndarray, a: float, theta: np.ndarray) -> np.ndarray:
    """Implements the normal-ogive (probit) item response theory (IRT) model.

    Parameters
    ----------
    b : np.ndarray
        Category thresholds (response options).
    a : float
        Item discrimination (slope).
    theta : np.ndarray
        Latent trait.

    Returns
    -------
    np.ndarray
        Response probabilities for the different options.
    """

    # Initialize list with response probabilities
    probs = []

    # Cycle over categories (choice options of questionnaire)
    for c in range(len(b) - 1):

        # Compute category probabilities
        pc = norm.cdf(a * (theta - b[c])) - norm.cdf(a * (theta - b[c + 1]))
        probs.append(pc)

    return np.array(probs)


def generate_questionnaire_data(
    n_subj: int, theta_sd: float = 0.25, eta_sd: float = 0.25, n_items: int = 200
) -> pd.DataFrame:
    """Simulate questionnaire data based on the normal-ogive model.

    Parameters
    ----------
    n_subj : int
        Number of subjects
    theta_sd : float
        Standard deviation of latent trait that is associated with the first questionnaire.
    eta_sd : float
        Standard deviation of latent trait that is associated with the second questionnaire.
    n_items : int
        Number of items.

    Returns
    -------
    pd.DataFrame
        Simulated questionnaire data.
    """

    # Initialize lists
    theta_list = list()
    eta_list = list()
    response_list = list()
    sub_list = list()
    id_list = list()
    item_list = list()
    questionnaire_list = list()

    # Cycle over subjects
    for s in range(n_subj):

        # Randomly sample theta and eta (latent traits) of current subject
        theta = np.random.normal(0, theta_sd)
        eta = np.random.normal(0, eta_sd)

        if s < 10:
            id = f"sim_00{s}"
        elif 10 <= s < 100:
            id = f"sim_0{s}"
        else:
            id = f"sim_{s}"

        # Cycle over items
        for i in range(n_items):

            # Randomly sample the item discrimination (slope)
            a = np.random.uniform(0.6, 1.0)

            # Randomly sample category thresholds
            b = np.random.uniform(-1.5, 1.5, 4)
            b = np.sort(b)
            b_ext = np.concatenate(([-np.inf], b, [np.inf]))

            # First half questionnaire 1, second half questionnaire 2
            if i <= np.round(n_items / 2):
                probs = normal_ogive(b_ext, a, theta)
                questionnaire = 1
            else:
                probs = normal_ogive(b_ext, a, eta)
                questionnaire = 2

            # Add results to lists
            response_list.append(np.random.choice(np.arange(1, 6), p=probs.flatten()))
            questionnaire_list.append(questionnaire)
            item_list.append(i)
            id_list.append(id)
            sub_list.append(s + 1)
            theta_list.append(theta)
            eta_list.append(eta)

    # Centrally store data
    questionnaire_data = pd.DataFrame(
        data={
            "subj_num": sub_list,
            "ID": id_list,
            "item": item_list,
            "response": response_list,
            "questionnaire": questionnaire_list,
            "theta": theta_list,
            "eta": eta_list,
        }
    )

    return questionnaire_data


def sample_hazard_rate(
    thetas: pd.Series, intercept: float = 0.0, slope: float = 2.0, sigma: float = 1.0
) -> np.ndarray:
    """Sample noisy hazard rates depending on latent trait theta.

    We first link theta and hazard rate based on a simple linear model and then use the inverse logit function
    to transform the value into the final form where h is \in [0, 1].

    Parameters
    ----------
    thetas : pd.Series
        Latent trait.
    intercept : float
        Intercept of the simple linear model.
    slope : float
        Slope of the linear model that links trait and noiseless hazard rate.
    sigma : float
        Gaussian standard deviation for noisy hazard rate.

    Returns
    -------
    np.ndarray
        Noisy hazard rate.
    """

    # Unconstrained hazard rate with correlation = 1
    unconstrained_hazard_rate = intercept + slope * np.array(thetas)

    # Add noise to the unconstrained hazard rate
    unconstrained_hazard_rate = np.random.normal(
        loc=unconstrained_hazard_rate, scale=sigma
    )

    # Use the inverse logit function to compute constrained hazard rate
    hazard_rate = 1 / (1 + np.exp(-unconstrained_hazard_rate))

    return hazard_rate

def simulate_and_run_regression(
    n_subj: int,
    hazard_rate: np.ndarray,
    df_exp: pd.DataFrame,
    sim: str | bool = "agent",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate data for the SCA using the RBM and run regression.

    Parameters
    ----------
    n_subj : int
        Number of subjects.
    hazard_rate : np.ndarray
        Sampled hazard rates.
    df_exp : pd.DataFrame
        Experimental data.
    sim : str | bool
        agent = simulate agent predictions based on empirical task data (default).
        task_agent = simulate agent predictions based on simulated task data.
        False = no simulation, only likelihood evaluation for empirical task data.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - Regression model results.
        - Simulated estimation errors.
        - Simulated data for SCA.
        - Simulated RBM data used for RBM estimation.
    """

    # -------------------------
    # Simulate data for the SCA
    # -------------------------

    gen_model = get_sim_params(n_subj, hazard_rate)

    # Run the simulation
    n_sim = 1  # 1 simulation per subject
    sim_est_errs, df_sim = simulation_loop(
        df_exp, gen_model, n_subj, plot_data=False, n_sim=n_sim, sim=sim
    )

    # Rename simulated variables to match experimental data conventions
    df_sim.rename(columns={"sim_a_t_rad": "a_t_rad"}, inplace=True)
    df_sim.rename(columns={"sim_b_t_rad": "b_t_rad"}, inplace=True)

    # Create dataframe for RBM estimation
    df_rbm = pd.DataFrame(index=range(0, len(df_sim)), dtype="float")
    df_rbm["subj_num"] = df_sim["subj_num"].copy()
    df_rbm["ID"] = df_sim["ID"].copy()
    df_rbm["new_block"] = df_sim["new_block"].copy()
    df_rbm["x_t_rad"] = df_sim["x_t_rad"].copy()
    df_rbm["a_t_rad"] = df_sim["a_t_rad"].copy()
    df_rbm["delta_t_rad"] = df_sim["delta_t_rad"].copy()
    df_rbm["v_t"] = df_sim["v_t"].copy()
    df_rbm["sigma"] = df_sim["sigma"].copy()
    df_rbm["mu_t_rad"] = df_sim["task_mu"].copy()
    df_rbm["b_t_rad"] = df_sim["b_t_rad"].copy()
    df_rbm["group"] = df_sim["group"].copy()

    # Noise-level dummy variable
    df_sim["kappa_dummy"] = np.nan
    df_sim.loc[df_sim["kappa"] == 8, "kappa_dummy"] = 1
    df_sim.loc[df_sim["kappa"] == 16, "kappa_dummy"] = -1

    # Hit dummy variable
    df_sim["hit_dummy"] = np.nan
    df_sim.loc[df_sim["hit"] == 1, "hit_dummy"] = 1
    df_sim.loc[df_sim["hit"] == 0, "hit_dummy"] = -1

    # Catch-trial dummy variable
    df_sim["v_dummy"] = np.nan
    df_sim.loc[df_sim["v_t"] == 1, "v_dummy"] = 1
    df_sim.loc[df_sim["v_t"] == 0, "v_dummy"] = -1

    # --------------------------------------------------------------------
    # Run the normative model to extract alpha for the regression analysis
    # --------------------------------------------------------------------

    # Normative simulation parameters
    norm_model = pd.DataFrame(
        columns=["omikron_0", "omikron_1", "lambda_0", "lambda_1", "h", "s", "u", "sigma_H", "subj_num"]
    )
    norm_model.loc[:, "omikron_0"] = np.repeat(1, n_subj)
    norm_model.loc[:, "omikron_1"] = np.repeat(0, n_subj)
    norm_model.loc[:, "lambda_0"] = np.repeat(-10, n_subj)
    norm_model.loc[:, "lambda_1"] = np.repeat(-0.5, n_subj)
    norm_model.loc[:, "h"] = np.repeat(0.1, n_subj)
    norm_model.loc[:, "s"] = np.repeat(1, n_subj)
    norm_model.loc[:, "u"] = np.repeat(0, n_subj)
    norm_model.loc[:, "sigma_H"] = np.repeat(0.001, n_subj)  # todo: run simulation with less impact of catch trials
    # We have to figure out if models that don't account for catch trials are dramatically worse.
    norm_model.loc[:, "subj_num"] = np.arange(n_subj) + 1

    # Run simulation
    _, df_norm = simulation_loop(
        df_sim, norm_model, n_subj, plot_data=False, n_sim=n_sim, sim=False
    )

    # Test if subject numbers still line up
    comp_subj_num = df_sim["subj_num"] == df_norm["subj_num"]
    if False in comp_subj_num.values:
        sys.exit("Sub IDs don't match!")

    # Create simulation dataframe for regression analysis
    df_sim_sca_data = pd.DataFrame(
        columns=[
            "a_t_rad",
            "delta_t_rad",
            "tau_t",
            "omega_t",
            "alpha_t",
            "hit_dummy",
            "e_t_rad",
            "kappa_dummy",
            "v_dummy",
            "v_t",
            "group",
            "subj_num",
            "ID",
        ]
    )

    # Create data frame with simulated data for regression analysis
    df_sim_sca_data["a_t_rad"] = df_sim["a_t_rad"].copy()
    df_sim_sca_data["delta_t_rad"] = df_sim["delta_t_rad"].copy()
    df_sim_sca_data["tau_t"] = df_norm["tau_t"].copy()
    df_sim_sca_data["omega_t"] = df_norm["omega_t"].copy()
    df_sim_sca_data["alpha_t"] = df_norm["alpha_t"].copy()
    df_sim_sca_data["e_t_rad"] = df_sim["sim_e_t_rad"].copy()
    df_sim_sca_data["v_t"] = df_sim["v_t"].copy()
    df_sim_sca_data["group"] = df_sim["group"].copy()
    df_sim_sca_data["subj_num"] = df_norm["subj_num"].copy()
    df_sim_sca_data["ID"] = df_norm["ID"].copy()
    df_sim_sca_data["hit_dummy"] = df_sim["hit_dummy"].copy()
    df_sim_sca_data["kappa_dummy"] = df_sim["kappa_dummy"].copy()
    df_sim_sca_data["v_dummy"] = df_sim["v_dummy"].copy()

    # Drop NaNs for regression
    df_sim_sca_data = df_sim_sca_data.dropna(
        subset=["delta_t_rad", "a_t_rad"]
    ).reset_index()

    # ---------------------------------
    # Run the standard regression model
    # ---------------------------------

    # Define regression variables
    # ---------------------------

    reg_vars = RegVars()
    reg_vars.n_subj = n_subj  # number of subjects
    reg_vars.n_ker = 4  # number of kernels for estimation
    reg_vars.n_sp = 5  # number of random starting points
    reg_vars.rand_sp = True  # use random starting points
    reg_vars.use_prior = False  # no prior for estimation

    # Fixed LR + adaptive LR + hit + noise
    # ------------------------------------

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: False,  # omega_t
        reg_vars.beta_3: False,  # tau_t
        reg_vars.beta_4: True,  # alpha_t
        reg_vars.beta_5: False,  # r_t
        reg_vars.beta_6: False,  # sigma_t
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

    # Initialize regression object instance
    for_regression = RegressionFor(reg_vars)

    # Run regression
    df_reg = for_regression.parallel_estimation(df_sim_sca_data, prior_columns)

    return df_reg, sim_est_errs, df_sim_sca_data, df_rbm


def get_sim_params(n_subj: int, hazard_rate: np.ndarray) -> pd.DataFrame:
    """Sample parameters for the RBM simulations.

    Parameters
    ----------
    n_subj : int
        Number of subjects.
    hazard_rate : np.ndarray
        Hazard rates.

    Returns
    -------
    pd.DataFrame
        Contains simulated parameters for the RBM simulations.

    """

    # Simulation parameters
    gen_model = pd.DataFrame(
        columns=["omikron_0", "omikron_1", "lambda_0", "lambda_1", "h", "s", "u", "sigma_H", "subj_num"]
    )

    # Create figure for parameter distributions
    plt.figure()

    # Omikron 0
    omikron_0 = np.random.normal(8, 3, size=n_subj)
    omikron_0[omikron_0 < 1] = 1
    gen_model.loc[:, "omikron_0"] = omikron_0

    plt.subplot(2, 2, 1)
    plt.hist(omikron_0, bins=20)
    plt.title(
        "Motor Noise: mean = "
        + str(np.round(np.mean(omikron_0), decimals=2))
        + ", SD = "
        + str(np.round(np.std(omikron_0), decimals=2))
    )

    # Omikron 1
    omikron_1 = np.random.normal(0.2, 0.2, size=n_subj)
    omikron_1[omikron_1 < 0] = 0
    omikron_1[omikron_1 > 1] = 1
    gen_model.loc[:, "omikron_1"] = omikron_1
    plt.subplot(2, 2, 2)
    plt.hist(omikron_1, bins=20)
    plt.title(
        "Learning-Rate Noise: mean = "
        + str(np.round(np.mean(omikron_1), decimals=2))
        + ", SD = "
        + str(np.round(np.std(omikron_1), decimals=2))
    )

    # Lambda 0
    lambda_0 = np.repeat(-10, n_subj)  # currently no perseveration
    gen_model.loc[:, "lambda_0"] = lambda_0

    # Lambda 1
    lambda_1 = np.repeat(-0.5, n_subj)  # currently no perseveration
    gen_model.loc[:, "lambda_1"] = lambda_1

    # Hazard rate based on latent factor
    gen_model.loc[:, "h"] = hazard_rate

    # Surprise sensitivity
    s = np.random.normal(0.4, 0.2, size=n_subj)
    s[s < 0] = 0
    s[s > 1] = 1
    gen_model.loc[:, "s"] = s
    plt.subplot(2, 2, 3)
    plt.hist(s, bins=20)
    plt.title(
        "Surprise Sensitivity: mean = "
        + str(np.round(np.mean(s), decimals=2))
        + ", SD = "
        + str(np.round(np.std(s), decimals=2))
    )

    # Uncertainty underestimation
    u = np.repeat(0, n_subj)
    gen_model.loc[:, "u"] = u
    plt.subplot(2, 2, 4)
    plt.hist(u, bins=20)
    plt.title(
        "Uncertainty Underestimation: mean = "
        + str(np.round(np.mean(u), decimals=2))
        + ", SD = "
        + str(np.round(np.std(u), decimals=2))
    )

    # Catch trials
    sigma_H = np.random.uniform(low=0.01, high=0.5, size=n_subj)
    gen_model.loc[:, "sigma_H"] = sigma_H

    plt.tight_layout()
    sns.despine()

    # Add subject numbers
    gen_model.loc[:, "subj_num"] = np.arange(n_subj) + 1  # data set starts with 1

    return gen_model
