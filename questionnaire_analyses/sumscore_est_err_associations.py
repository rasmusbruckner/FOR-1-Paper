"""Plot correlations between estimation errors and questionnaire scores."""

if __name__ == "__main__":

    import os
    import platform

    import matplotlib
    from allinpy import latex_plt

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    # Simple cross-platform backend selection
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")  # Headless
    elif platform.system() == "Darwin":
        matplotlib.use("MacOSX")  # macOS native
    else:
        matplotlib.use("Qt5Agg")

    import matplotlib.pyplot as plt
    import pandas as pd

    from FOR_1_Paper.for_utilities import (
        plot_main_questionnaire_correlations,
        plot_questionnaire_correlations_noise)

    # Questionnaires
    df_questionnaires = pd.read_pickle("for_data/questionnaire_sumscores.pkl")
    df_questionnaires = df_questionnaires.sort_values(by=["subj_num"])

    # Load data excluding change points for estimation errors
    df_for = pd.read_pickle("for_data/data_prepr_model.pkl")
    no_cp = df_for["c_t"].to_numpy() == 0
    df_for = df_for[no_cp]

    # Sort values just to be sure
    df_for = df_for.sort_values(by=["subj_num"])

    # Get IDs in both regression and questionnaire scores (where some filled out Qs incompletely)
    common_ids = set(df_for["ID"]) & set(df_questionnaires["subj_num"])

    # Filter data frames
    df_for = df_for[df_for["ID"].isin(common_ids)].reset_index(drop=True)

    # Compute estimation error for each subject
    mean_per_subject = df_for.groupby("subj_num")["e_t"].mean().reset_index()
    mean_per_subject = mean_per_subject.reset_index(drop=True)

    # Compute estimation error for noise conditions
    mean_per_subject_noise = (
        df_for.groupby(["subj_num", "kappa_t"])["e_t"].mean().reset_index()
    )
    mean_per_subject_low_noise = mean_per_subject_noise[
        mean_per_subject_noise["kappa_t"] == 16
    ].reset_index(drop=True)
    mean_per_subject_high_noise = mean_per_subject_noise[
        mean_per_subject_noise["kappa_t"] == 8
    ].reset_index(drop=True)
    mean_per_subject_low_noise = mean_per_subject_low_noise.reset_index(drop=True)
    mean_per_subject_high_noise = mean_per_subject_high_noise.reset_index(drop=True)

    # -----------------
    # Plot correlations
    # -----------------

    # Turn interactive mode on for plotting in debugger on Linux
    plt.ion()

    # 1. Plot CAPE, IUS, SPQ
    # ---------------------

    var = "e_t"
    ylabel = "Estimation Error"
    use_corr = False
    plot_main_questionnaire_correlations(
        df_questionnaires, mean_per_subject, var, ylabel, use_corr=use_corr
    )
    savename = "figures/main_questionnaires_est_err.png"
    plt.savefig(savename, transparent=False, dpi=400)

    # 2. Fixed learning rate high vs low noise
    # ----------------------------------------

    plot_questionnaire_correlations_noise(
        df_questionnaires,
        mean_per_subject_low_noise,
        mean_per_subject_high_noise,
        var,
        ylabel,
        use_corr=use_corr,
    )
    savename = "figures/IUS_SPQ_low_high_noise_est_err.png"
    plt.savefig(savename, transparent=False, dpi=400)

    # Show plot
    plt.ioff()
    plt.show()
