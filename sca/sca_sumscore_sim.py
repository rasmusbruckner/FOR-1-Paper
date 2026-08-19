"""SCA analysis for simulated data."""

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
        matplotlib.use("Qt5Agg")  # Linux with display, Windows, others

    import os

    import matplotlib.pyplot as plt
    import pandas as pd
    from SpecificationCurveAnalysis import sca_wrapper

    # Enable interactive mode for debugging
    plt.ion()

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # Rerun all models
    force_rerun = False

    # ---------
    # Load data
    # ---------

    # SCA folder
    sca_folder = "sca_task_agent_N200_T400_seed123/"

    # Questionnaire data
    df_questionnaires = pd.read_pickle(
        "for_data/"
        + sca_folder
        + "sim_sca_sum_scores_q1_seed123_N200_T400_task_agent.pkl"
    )
    df_questionnaires = df_questionnaires.sort_values(by=["subj_num"])

    # Regression data
    df_for = pd.read_pickle(
        "for_data/" + sca_folder + "sim_sca_regression_seed123_N200_T400_task_agent.pkl"
    )
    df_for = df_for.dropna(subset=["delta_t_rad", "a_t_rad"]).reset_index()  # drop nans

    # RBM data
    df_rbm = pd.read_pickle(
        "for_data/" + sca_folder + "sim_sca_rbm_seed123_N200_T400_task_agent.pkl"
    )

    # -------------------------------------
    # First questionnaire (positive result)
    # -------------------------------------

    # Run SCA
    sca_wrapper(
        sca_folder,
        df_questionnaires,
        df_for,
        df_rbm,
        which_var_quest=["response"],
        force_rerun=force_rerun,
        which_analysis_str="sum_score_sim_pos",
    )

    # Save figure
    save_name = (
        "/"
        + home_dir
        + "/rasmus/Dropbox/Apps/Overleaf/FOR-1-Paper/Figures/for_figure_4.pdf"
    )
    plt.savefig(save_name, dpi=400)

    # ----------------------------------
    # Second questionnaire (null result)
    # ----------------------------------

    # Questionnaire data
    df_questionnaires = pd.read_pickle(
        "for_data/"
        + sca_folder
        + "sim_sca_sum_scores_q2_seed123_N200_T400_task_agent.pkl"
    )
    df_questionnaires = df_questionnaires.sort_values(by=["subj_num"])

    # Run SCA
    sca_wrapper(
        sca_folder,
        df_questionnaires,
        df_for,
        df_rbm,
        which_var_quest=["response"],
        show_validation=False,
        force_rerun=force_rerun,
        which_analysis_str="sum_score_sim_null",
    )

    # Save figure
    save_name = (
        "/"
        + home_dir
        + "/rasmus/Dropbox/Apps/Overleaf/FOR-1-Paper/Figures/for_figure_5.pdf"
    )
    plt.savefig(save_name, dpi=400)

    # Show plot
    plt.ioff()
    plt.show()
