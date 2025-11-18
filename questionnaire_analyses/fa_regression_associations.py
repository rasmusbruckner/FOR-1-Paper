"""Single factor analysis results."""

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
        matplotlib.use("Qt5Agg")

    import matplotlib.pyplot as plt
    import seaborn as sns
    from allinpy import cm2inch, latex_plt

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    import os

    import pandas as pd

    from FOR_1_Paper.for_utilities import plot_correlation
    from FOR_1_Paper.sca.sca_utils import correlate_reg_fa, filter_subjects

    # ----------
    # Load data
    # ---------
    # Parameter space of factor analysis
    param_space = {
        "analysis_type": ["simple", "bifactor"],
        "data_type": ["big_data"],
        "n_factors": [2, 3, 4, 5],
        "rotation": ["oblimin", "varimax"],  # "cluster",
        "factor_method": ["minres", "ml"],
        "threshold_loadings": [True],
        "threshold_value": [0.2],
        "fs_method": ["Anderson", "Bartlett"],
        "fs_impute": ["mean"],
        "n_observations": [633],
        "n_variables": [118],
    }

    # Choose specific specifications for the current plots
    analysis_spec = {
        "fa": {
            "analysis_type": "simple",
            "data_type": "big_data",
            "factor_method": "minres",
            "fs_impute": "mean",
            "fs_method": "Anderson",
            "n_factors": 2,
            "n_observations": 633,
            "n_variables": 118,
            "rotation": "oblimin",
            "threshold_loadings": True,
            "threshold_value": 0.2,
        },
        "psychosis": True,
    }

    # Load data of best regression model
    df_regression = pd.read_pickle("for_data/regression_model_2_3_50_sp.pkl")

    # Load data for regression applied separately for low and high noise
    df_regression_low_noise = pd.read_pickle(
        "for_data/regression_model_low_noise_2_3_50_sp.pkl"
    )
    df_regression_high_noise = pd.read_pickle(
        "for_data/regression_model_high_noise_2_3_50_sp.pkl"
    )
    # Filter datasets to ensure they match
    df_sca_fa, df_reg, fa_hash = filter_subjects(analysis_spec, df_regression)

    # Estimation error
    # ----------------
    # Todo: we will integrate EE into the SCA. Then, we can use shared functions

    # Load data excluding change points for estimation errors
    df_for = pd.read_pickle("for_data/data_prepr_model.pkl")
    no_cp = df_for["c_t"].to_numpy() == 0
    df_for = df_for[no_cp]

    # Sort values just to be sure
    df_for = df_for.sort_values(by=["subj_num"])

    # Get IDs in both regression and questionnaire scores (where some filled out Qs incompletely)
    common_ids = set(df_for["ID"]) & set(df_sca_fa["ID"])

    # Filter data frames
    df_for = df_for[df_for["ID"].isin(common_ids)].reset_index(drop=True)

    # Compute estimation error for each subject
    mean_per_subject = df_for.groupby("subj_num")["e_t"].mean().reset_index()
    mean_per_subject = mean_per_subject.reset_index(drop=True)

    # ----------------------------
    # Plot factor analysis results
    # ----------------------------

    # Fixed learning rate
    # -------------------

    # We are using the factor with the highest correlation with CAPE
    which_factor = pd.read_pickle("for_data/which_factor.pkl")
    factor_name = which_factor[which_factor["fa_hash"] == fa_hash]["max_index"].iloc[0]

    # Select fixed learning rate
    which_var = "beta_1"

    # Enable interactive mode for debugging
    plt.ion()

    # Figure size
    fig_height = 5
    fig_width = 12

    # Create figure
    plt.figure(figsize=cm2inch(fig_width, fig_height))

    plt.subplot(131)
    plot_correlation(
        df_sca_fa,
        df_reg[[which_var]],
        factor_name,
        factor_name,
        "Fixed Learning Rate",
    )

    # Check if SCA would compute the same correlation
    analysis_result_fixed_lr, _, _, _ = correlate_reg_fa(
        which_factor, fa_hash, df_sca_fa, df_reg, which_var
    )
    print(analysis_result_fixed_lr)

    # Adaptive learning rate
    # ----------------------

    # Select adaptive learning rate
    which_var = "beta_4"

    plt.subplot(132)
    plot_correlation(
        df_sca_fa,
        df_reg[[which_var]],
        factor_name,
        factor_name,
        "Adaptive Learning Rate",
    )
    # Check if SCA would compute the same correlation
    analysis_result_fixed_lr, _, _, _ = correlate_reg_fa(
        which_factor, fa_hash, df_sca_fa, df_reg, which_var
    )
    print(analysis_result_fixed_lr)

    # Estimation error
    # ----------------
    plt.subplot(133)
    plot_correlation(
        df_sca_fa,
        mean_per_subject[["e_t"]],
        factor_name,
        factor_name,
        "Estimation Error",
    )

    plt.tight_layout()
    sns.despine()

    # Save figure
    savename = "figures/fa_analysis_" + fa_hash + ".png"
    plt.savefig(savename, transparent=False, dpi=400)

    # Show plot
    plt.ioff()
    plt.show()
