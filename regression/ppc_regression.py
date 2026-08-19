"""Posterior predictive check for regression model.

Todo: check which model version we ultimately want to use for paper.
    Either most complex or best fitting one.
"""

if __name__ == "__main__":

    import os

    import matplotlib

    # Use preferred backend for Linux, or just take default
    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        pass

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import seaborn as sns
    from allinpy import cm2inch, latex_plt
    from ForRegVars import RegVars
    from RegressionFor import RegressionFor
    from tqdm import tqdm

    from FOR_1_Paper.for_utilities import safe_save_dataframe

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    # Turn interactive mode on
    plt.ion()

    # Set random number generator for reproducible results
    seed = 123
    np.random.seed(seed)

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # Figure properties
    fig_height = 6
    fig_width = 6

    # ------------
    # 1. Load data
    # ------------

    df_for = pd.read_pickle("for_data/data_prepr_model.pkl")  # behavioral data
    n_subj = len(np.unique(df_for["subj_num"]))  # number of subjects
    n_trials = len(np.unique(df_for["trial"]))  # number of trials
    df_reg = pd.read_pickle("for_data/regression_23_50sp.pkl")  # regression parameters

    # ----------------
    # 2. Simulate data
    # ----------------

    # Initialize regression variables
    reg_vars = RegVars()
    reg_vars.n_subj = n_subj
    reg_vars.use_prior = False

    # Determine which parameters should be used for simulation
    reg_vars.which_vars = {
        "beta_0": True,  # intercept
        "beta_1": True,  # PE (fixed learning rate)
        "beta_2": False,  # interaction PE and RU
        "beta_3": False,  # interaction PE and CPP
        "beta_4": True,  # interaction PE and alpha
        "beta_5": True,  # interaction PE and hit
        "beta_6": True,  # interaction PE and noise condition
        "beta_7": True,  # interaction PE and visible
        "beta_8": False,  # interaction EE and visible
        "omikron_0": True,  # motor noise (independent of UP)
        "omikron_1": True,  # learning-rate noise (dependent on UP)
        "lambda_0": False,  # perseveration intercept
        "lambda_1": False,  # perseveration slope
    }

    # Simulation parameters
    df_model = pd.DataFrame(
        columns=[
            "beta_0",
            "beta_1",
            "beta_4",
            "beta_5",
            "beta_6",
            "omikron_0",
            "omikron_1",
            "subj_num",
        ]
    )
    df_model.loc[:, "subj_num"] = df_reg["subj_num"].to_numpy()
    df_model.loc[:, "beta_0"] = df_reg["beta_0"].to_numpy()
    df_model.loc[:, "beta_1"] = df_reg["beta_1"].to_numpy()
    df_model.loc[:, "beta_4"] = df_reg["beta_4"].to_numpy()
    df_model.loc[:, "beta_5"] = df_reg["beta_5"].to_numpy()
    df_model.loc[:, "beta_6"] = df_reg["beta_6"].to_numpy()
    df_model.loc[:, "omikron_0"] = df_reg["omikron_0"].to_numpy()
    df_model.loc[:, "omikron_1"] = df_reg["omikron_1"].to_numpy()

    # Parameters to include for simulation
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # delta_t
        reg_vars.beta_2: False,  # omega_t
        reg_vars.beta_3: False,  # tau_t
        reg_vars.beta_4: True,  # alpha_t
        reg_vars.beta_5: True,  # r_t
        reg_vars.beta_6: True,  # sigma_t
        reg_vars.beta_7: False,  # catch-trial * PE
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

    # Create regression-object instance
    regression = RegressionFor(reg_vars)

    # Simulate updates based on sampled parameters
    n_sims = 100  # number of simulations

    # Initialize progress bar
    pbar = tqdm(total=n_sims)

    # Initialize model-data array
    mean_a_t_model = np.full([n_subj, n_sims], np.nan)

    # Generate model predictions
    for i in range(n_sims):

        # Sample parameters
        samples = regression.sample_data(df_model, n_trials, df_for)

        # Compute mean in degrees
        mean_a_t_model[:, i] = np.rad2deg(
            samples.groupby("subj_num")["a_t_rad"].apply(lambda x: np.mean(np.abs(x)))
        )

        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    # Save as data frame
    mean_a_t_model = pd.DataFrame(mean_a_t_model)
    mean_a_t_model.name = "regression_ppc_seed_" + str(seed)
    safe_save_dataframe(mean_a_t_model)

    # ----------------------------------
    # 3. Plot posterior predictive check
    # ----------------------------------

    # Drop nans in empirical data for consistency with simulation data
    df_for = df_for.dropna(subset=["delta_t_rad"]).reset_index(drop=True)

    # Compute mean in degrees of empirical data
    mean_a_t_sub = df_for.groupby("subj_num")["a_t"].apply(lambda x: np.mean(np.abs(x)))

    # Compute model mean across all simulations: this will be the dot in our PPC plot
    grand_mean_model = mean_a_t_model.mean(axis=1)

    # Create figure
    f = plt.figure(figsize=cm2inch(fig_width, fig_height))

    # Plot diagonal line
    plt.axline((0, 0), slope=1)

    # Plot line between min and max values of the simulations
    plt.vlines(
        x=mean_a_t_sub,
        ymin=mean_a_t_model.min(axis=1),
        ymax=mean_a_t_model.max(axis=1),
        zorder=0,
    )

    # Add mean model and empirical data
    plt.scatter(mean_a_t_sub, grand_mean_model, marker=".", color="k", zorder=100)

    # Plot styling
    r, _ = stats.spearmanr(mean_a_t_sub, grand_mean_model)
    plt.title(f"$r=${round(r, 2)}")
    plt.xlim(8, 38)
    plt.ylim(8, 38)
    plt.xlabel("Mean update participants")
    plt.ylabel("Mean update model")
    sns.despine()
    plt.tight_layout()

    # Save figure
    # -----------

    save_name = (
        "/"
        + home_dir
        + "/rasmus/Dropbox/Apps/Overleaf/FOR-1-Paper/Figures/regression_ppc.pdf"
    )
    plt.savefig(save_name, dpi=400)

    # ---------------------------
    # 4. Plot individual subjects
    # ---------------------------

    # Cycle over subjects
    for i in range(n_subj):

        # Plot subjects
        plt.figure()
        plt.scatter(
            df_for[df_for["subj_num"] == i + 1]["delta_t_rad"],
            df_for[df_for["subj_num"] == i + 1]["a_t_rad"],
        )

        # Plot last model simulation
        plt.scatter(
            samples[samples["subj_num"] == i + 1]["delta_t_rad"],
            samples[samples["subj_num"] == i + 1]["a_t_rad"],
        )

        # Plot styling
        plt.xlabel("Prediction error")
        plt.ylabel("Update")
        plt.legend(["Data", "Model"])

        # Save the plot
        savename = "figures/single_trial/regression/up_%s.pdf" % i
        plt.savefig(savename)
        plt.close()

    plt.ioff()
    plt.show()
