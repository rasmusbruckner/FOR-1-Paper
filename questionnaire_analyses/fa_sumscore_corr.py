"""Compute correlation matrix between factor analysis scores and questionnaire scores.

Also store which factor is the most correlated with CAPE_pos for SCA.
"""

if __name__ == "__main__":

    import os
    import platform

    import matplotlib

    system = platform.system()

    # Simple cross-platform backend selection
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")  # Headless
    elif platform.system() == "Darwin":
        matplotlib.use("MacOSX")  # macOS native
    else:
        matplotlib.use("Qt5Agg")

    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import seaborn as sns
    from allinpy import cm2inch
    from tqdm import tqdm

    from FOR_1_Paper.for_utilities import safe_save_dataframe
    from FOR_1_Paper.sca.sca_utils import (build_specs_with_vars,
                                           fa_candidates, filter_subjects)

    # ------------
    # 1. Load data
    # ------------

    df_for = pd.read_pickle("for_data/data_prepr_model.pkl")
    df_for = df_for.dropna(subset=["delta_t_rad", "a_t_rad"]).reset_index()  # drop nans
    n_subj = len(np.unique(df_for["subj_num"]))  # number of subjects

    # SCA folder
    folder = Path("for_data")

    df_questionnaires = pd.read_pickle("for_data/questionnaire_sumscores.pkl")
    df_questionnaires = df_questionnaires.sort_values(by=["subj_num"])

    # ------------------------------
    # Factor analysis specifications
    # ------------------------------

    # Parameter space of factor analysis
    param_space = {
        "analysis_type": ["simple", "bifactor"],
        "data_type": ["big_data"],
        "n_factors": [2, 3, 4, 5],
        "rotation": ["oblimin", "cluster", "varimax"],
        "factor_method": ["minres", "ml"],
        "threshold_loadings": [True],
        "threshold_value": [0.2],
        "fs_method": ["Anderson", "Bartlett"],
        "fs_impute": ["mean"],
        "n_observations": [633],
        "n_variables": [118],
    }

    # Analysis specifications based on our parameter space
    pool = list(fa_candidates(param_space))

    # Take into account all factors
    # -----------------------------

    var_rule_ids = ["any_factor"]
    var_rules = {
        "any_factor": [
            lambda fa: True,  # always applies
        ],
    }

    # Create analysis specifications based on pool and variable rules
    analysis_specs = build_specs_with_vars(pool, var_rule_ids, var_rules)

    # ------------------------
    # Run correlation analyses
    # ------------------------

    # Ensure that we have an empty folder w/o any previous results
    os.makedirs("for_data/sca", exist_ok=True)
    for f in os.listdir("for_data/sca/"):
        os.remove(os.path.join("for_data/sca/", f))

    # Initialize progress bar
    pbar = tqdm(total=len(analysis_specs.items()))

    # Initialize lists
    hash_list = list()
    max_value_list = list()
    max_index_list = list()

    # Cycle over all factor analysis specifications and compute correlation
    # ---------------------------------------------------------------------

    for analysis_name, analysis_spec in analysis_specs.items():

        # Rename subject column
        df_questionnaires = df_questionnaires.rename(columns={"subj_num": "ID"})

        # Filter subjects
        df_factor_analysis, df_questionnaires, fa_hash = filter_subjects(
            analysis_spec, df_questionnaires
        )

        # Specify which factors to exclude from correlation matrix
        exclude_factors = [
            "CAPE1",
            "CAPE2",
            "IUS",
            "age",
            "gender",
            "G_small",
            "F1_small",
            "F2_small",
            "G_big",
            "F1_big",
            "F2_big",
        ]

        # Get factor columns (all except ID)
        factor_cols = [col for col in df_factor_analysis.columns if col != "ID"]

        # Get questionnaire columns (all except 'ID')
        questionnaire_cols = [
            col
            for col in df_questionnaires.columns
            if col != "ID" and col not in exclude_factors
        ]

        # Calculate correlation matrix
        # ----------------------------

        # Initialize correlation matrix
        correlation_matrix = pd.DataFrame(
            index=questionnaire_cols, columns=factor_cols, dtype=float
        )

        # Fill correlation matrix
        for q_col in questionnaire_cols:
            for f_col in factor_cols:
                r, _ = stats.pearsonr(
                    df_questionnaires[q_col].to_numpy(),
                    df_factor_analysis[f_col].to_numpy(),
                )
                correlation_matrix.loc[q_col, f_col] = r

        # Convert to numeric
        correlation_matrix = correlation_matrix.astype(float)

        # Plot correlation matrix
        fig_corr, ax_corr = plt.subplots(figsize=(cm2inch(20, 15)))

        sns.heatmap(
            correlation_matrix,
            ax=ax_corr,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Pearson r"},
            square=False,
        )

        # Add labels
        ax_corr.set_xlabel("Factor Analysis Scores")
        ax_corr.set_ylabel("Questionnaire Scores")
        ax_corr.set_title(f"Correlation Matrix (FA: {analysis_name})")
        plt.tight_layout()

        # Save correlation matrix plot
        corr_output_file = (
            f"figures/correlation matrix/correlation_matrix_{fa_hash}.png"
        )
        plt.savefig(corr_output_file, dpi=300, bbox_inches="tight")
        plt.close(fig_corr)

        # Store which factor is the most correlated with CAPE_pos
        max_index = abs(correlation_matrix.loc["CAPE_pos", :]).idxmax()
        max_value = correlation_matrix.loc["CAPE_pos", max_index]
        max_index_list.append(max_index)
        max_value_list.append(max_value)
        hash_list.append(fa_hash)

        # Update progress bar
        pbar.update()

    # Close progress bar
    pbar.close()

    # Save which factor is the most correlated with CAPE_pos
    which_factor = pd.DataFrame(
        data={
            "fa_hash": hash_list,
            "max_index": max_index_list,
            "max_value": max_value_list,
        }
    )
    which_factor.name = f"which_factor"
    safe_save_dataframe(which_factor)
