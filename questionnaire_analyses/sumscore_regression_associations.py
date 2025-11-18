"""Plot correlations between questionnaire scores and regression model parameters."""

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
    from allinpy import latex_plt

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    import os

    import pandas as pd

    from FOR_1_Paper.for_utilities import (
        plot_idas_correlations, plot_main_questionnaire_correlations,
        plot_questionnaire_correlations_noise)

    # ----------
    # Load data
    # ---------
    # Questionnaires
    df_questionnaires = pd.read_pickle("for_data/questionnaire_sumscores.pkl")
    df_questionnaires = df_questionnaires.sort_values(by=["subj_num"])

    # Load data of best regression model
    df_regression = pd.read_pickle("for_data/regression_model_2_3_50_sp.pkl")

    # Load data for regression applied separately for low and high noise
    df_regression_low_noise = pd.read_pickle(
        "for_data/regression_model_low_noise_2_3_50_sp.pkl"
    )
    df_regression_high_noise = pd.read_pickle(
        "for_data/regression_model_high_noise_2_3_50_sp.pkl"
    )

    # Sort values just to be sure
    df_regression = df_regression.sort_values(by=["subj_num"])
    df_regression_low_noise = df_regression_low_noise.sort_values(by=["subj_num"])
    df_regression_high_noise = df_regression_high_noise.sort_values(by=["subj_num"])

    # Get IDs in both regression and questionnaire scores (where some filled out Qs incompletely)
    common_ids = set(df_regression["ID"]) & set(df_questionnaires["subj_num"])

    # Filter data frames
    df_regression = df_regression[df_regression["ID"].isin(common_ids)].reset_index(
        drop=True
    )
    df_regression_low_noise = df_regression_low_noise[
        df_regression_low_noise["ID"].isin(common_ids)
    ].reset_index(drop=True)
    df_regression_high_noise = df_regression_high_noise[
        df_regression_high_noise["ID"].isin(common_ids)
    ].reset_index(drop=True)
    df_factor_analysis = df_questionnaires[
        df_questionnaires["subj_num"].isin(common_ids)
    ].reset_index(drop=True)

    # -------------------
    # Plot CAPE, IUS, SPQ
    # -------------------

    # Enable interactive mode for debugging
    plt.ion()

    # Fixed learning rate
    # -------------------

    var = "beta_1"
    ylabel = "Fixed Learning Rate"
    use_corr = False
    plot_main_questionnaire_correlations(
        df_questionnaires, df_regression, var, ylabel, use_corr=use_corr
    )
    savename = "figures/main_questionnaires_fixedLR.png"
    plt.savefig(savename, transparent=False, dpi=400)

    # Adaptive learning rate
    # ----------------------

    var = "beta_4"
    ylabel = "Adaptive Learning Rate"
    plot_main_questionnaire_correlations(
        df_questionnaires, df_regression, var, ylabel, use_corr=use_corr
    )
    savename = "figures/main_questionnaires_adaptiveLR.png"
    plt.savefig(savename, transparent=False, dpi=400)

    # Fixed learning rate high vs low noise
    # -------------------------------------

    var = "beta_1"
    ylabel = "Fixed Learning Rate"
    plot_questionnaire_correlations_noise(
        df_questionnaires,
        df_regression_low_noise,
        df_regression_high_noise,
        var,
        ylabel,
        use_corr=use_corr,
    )
    savename = "figures/IUS_SPQ_low_high_noise_fixedLR.png"
    plt.savefig(savename, transparent=False, dpi=400)

    # Adaptive learning rate high vs low noise
    # ----------------------------------------

    var = "beta_4"
    ylabel = "Adaptive Learning Rate"
    plot_questionnaire_correlations_noise(
        df_questionnaires,
        df_regression_low_noise,
        df_regression_high_noise,
        var,
        ylabel,
        use_corr=use_corr,
    )
    savename = "figures/IUS_SPQ_low_high_noise_adaptiveLR.png"
    plt.savefig(savename, transparent=False, dpi=400)

    # --------------
    # IDAS subscales
    # --------------

    # Fixed learning rate
    # -------------------

    var = "beta_1"
    ylabel = "Fixed Learning Rate"
    plot_idas_correlations(
        df_questionnaires, df_regression, var, ylabel, use_corr=use_corr
    )
    savename = "figures/IDAS_fixedLR.png"
    plt.savefig(savename, transparent=False, dpi=400)

    # Adaptive learning rate
    # ----------------------

    var = "beta_4"
    ylabel = "Adaptive Learning Rate"
    plot_idas_correlations(
        df_questionnaires, df_regression, var, ylabel, use_corr=use_corr
    )
    savename = "figures/IDAS_adaptiveLR.png"
    plt.savefig(savename, transparent=False, dpi=400)

    # Show plot
    plt.ioff()
    plt.show()
