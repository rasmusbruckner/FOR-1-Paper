"""SCA analysis for empirical research unit data set."""

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
    import numpy as np
    import pandas as pd
    from SpecificationCurveAnalysis import sca_wrapper

    # Enable interactive mode for debugging
    plt.ion()

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    force_rerun = False

    # ---------
    # Load data
    # ---------

    df_questionnaires = pd.read_pickle("for_data/questionnaire_sumscores.pkl")
    df_questionnaires = df_questionnaires.sort_values(by=["subj_num"])
    n_subj_quest = len(np.unique(df_questionnaires["subj_num"]))

    df_for = pd.read_pickle("for_data/data_prepr_model.pkl")
    df_for = df_for.dropna(subset=["delta_t_rad", "a_t_rad"]).reset_index()
    n_subj_reg = len(np.unique(df_for["subj_num"]))

    df_rbm = pd.read_pickle("for_data/data_prepr.pkl")
    n_subj_rbm = len(np.unique(df_rbm["subj_num"]))

    # SCA folder
    sca_folder = "sca_for/"

    # Run SCA
    sca_wrapper(
        sca_folder,
        df_questionnaires,
        df_for,
        df_rbm,
        which_var_quest=["CAPE1", "SPQ1"],
        force_rerun=force_rerun,
        which_analysis_str="sum_score",
    )

    # Save figure
    save_name = (
        "/"
        + home_dir
        + "/rasmus/Dropbox/Apps/Overleaf/FOR-1-Paper/Figures/for_figure_8.pdf"
    )
    plt.savefig(save_name, dpi=400)

    # Show plot
    plt.ioff()
    plt.show()
