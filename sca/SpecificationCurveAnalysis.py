from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import pickle
import re
from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
import scipy
import scipy.stats as stats
import seaborn as sns
from allinpy import cm2inch
from matplotlib.colors import ListedColormap
from numpy.random import default_rng
from rbmpy import AgentVars
from tqdm import tqdm

from FOR_1_Paper.for_utilities import safe_save_dataframe
from FOR_1_Paper.modeling.ForEstimation import ForEstimation
from FOR_1_Paper.modeling.ForEstVars import ForEstVars
from FOR_1_Paper.regression.ForRegVars import RegVars
from FOR_1_Paper.regression.RegressionFor import RegressionFor


class SpecificationCurveAnalysis:
    """Specifies the instance variables and methods of the specification curve analysis class.

    Attributes
    ----------
    n_subj : int | None
        Number of subjects used for analysis.
    pool : list[dict[str, str]] | None
           List of dictionaries containing FA data used to generate hash codes.
    sca_fa_folder : str
        Folder with SCA factor analyses
    sca_fa_filetype : str
        File type for SCA factor analyses (.pkl or .csv)
    """

    n_subj: int | None
    pool: list[dict[str, str]] | None
    sca_fa_folder: str
    sca_fa_filetype: str
    which_analysis_str: str
    var_names: list | None
    var_rules: dict | None
    expected_n_subj: int

    significance_counter: int | None
    counter: int | None
    all_quest_data: list | None
    all_parameters: list | None
    all_results: list | None

    def __init__(self):
        """Defines the instance variables unique to each instance."""

        self.n_subj = None
        self.pool = None
        self.sca_fa_folder = "for_data"
        self.sca_fa_filetype = ".pkl"
        self.which_analysis_str = "fixed_LR"
        self.var_names = None
        self.var_rules = None
        self.expected_n_subj = 65
        self.significance_counter = None
        self.counter = None
        self.all_quest_data = None
        self.all_parameters = None
        self.all_results = None

    def run_sca(
        self,
        regression_specs: dict,
        reg_vars: RegVars,
        rbm_specs: dict,
        est_vars: ForEstVars,
        agent_vars: AgentVars,
        df_for: pd.DataFrame,
        df_rbm: pd.DataFrame,
        df_questionnaires: pd.DataFrame,
        which_var_quest: list,
        force_rerun=False,
    ) -> None:
        """Run specification curve analysis (SCA) for multiple regression, RBM, and questionnaire specifications.

        Parameters
        ----------
        regression_specs : dict
            Dictionary containing regression specifications, where each key specifies
            a regression name and its corresponding value is a specification dictionary.
        reg_vars : RegVars
            A custom class/object containing variables and settings necessary for executing
            regression models.
        rbm_specs : dict
            Dictionary containing RBM specifications. Each key specifies an RBM type and its configuration.
        est_vars : EstVars
            A custom class/object containing variables and settings necessary for estimating the RBM.
        agent_vars : AgentVars
            A custom class/object containing RBM agent variables.
        df_for : pd.DataFrame
            Data frame containing input required for regression analysis.
        df_rbm : pd.DataFrame
            Data frame containing input required for RBM analysis.
        df_questionnaires : pd.DataFrame
            Data Frame containing questionnaire scores.
        which_var_quest : list
            Questionnaire variable(s) of interest
        force_rerun : bool, optional
            Determines whether to force re-running of regression models, by default False.

        Returns
        -------
        None
            This function does not return any value.
        """

        # Initialize output lists
        self.all_quest_data = []
        self.all_parameters = []
        self.all_results = []

        # Initialize counter for significant results
        self.significance_counter = 0

        # Initialize counter for specifications
        self.counter = 0

        # Cycle over all specifications and compute correlation
        # -----------------------------------------------------

        # Regression specifications
        for reg_name, reg_spec in regression_specs.items():

            # Dependent variable (currently fixed or adaptive LR)
            for dep_name, dep_spec in reg_spec["dependent_variable"].items():

                if dep_spec:

                    # Compute or load regression result (if pre-computed)
                    df_reg = run_or_load_regression(
                        reg_name,
                        reg_spec[reg_name],
                        reg_vars,
                        df_for,
                        force_rerun=force_rerun,
                    )

                    # Flip beta_4 (adaptive LR) to ensure all effects go in same direction:
                    # Naturally, higher beta_4 is more adaptive, but beta_1 is the other way around.
                    # We flip beta_4 to ensure all effects are consistent.
                    if dep_name == "beta_4":
                        df_reg["beta_4"] *= -1

                    # Questionnaire variable
                    for quest in which_var_quest:

                        # Compute correlation
                        self.counter += 1
                        self.compute_sca_correlations(
                            df_questionnaires,
                            df_reg,
                            dep_name,
                            quest,
                            reg_spec[reg_name],
                        )

        # RBM specifications
        for rbm_name, rbm_spec in rbm_specs.items():

            # Compute or load regression result (if pre-computed)
            df_model = run_or_load_rbm(
                rbm_name,
                rbm_spec,
                est_vars,
                agent_vars,
                df_rbm,
                force_rerun=force_rerun,
            )

            # We currently on have hazard rate for RBM
            which_var_rbm = "h"

            # Questionnaire variable
            for quest in which_var_quest:

                # Compute correlation
                self.counter += 1
                self.compute_sca_correlations(
                    df_questionnaires, df_model, which_var_rbm, quest, rbm_spec
                )

        print(f"Number of significant results: {self.significance_counter}")

    def run_sca_vaghi(
        self,
        regression_specs: dict,
        reg_vars: RegVars,
        rbm_specs: dict,
        est_vars: ForEstVars,
        agent_vars: AgentVars,
        df_for: pd.DataFrame,
        df_rbm: pd.DataFrame,
        force_rerun=False,
    ) -> None:
        """Run specification curve analysis (SCA) for multiple regression, RBM, and questionnaire specifications.

        Parameters
        ----------
        regression_specs : dict
            Dictionary containing regression specifications, where each key specifies
            a regression name and its corresponding value is a specification dictionary.
        reg_vars : RegVars
            A custom class/object containing variables and settings necessary for executing
            regression models.
        rbm_specs : dict
            Dictionary containing RBM specifications. Each key specifies an RBM type and its configuration.
        est_vars : EstVars
            A custom class/object containing variables and settings necessary for estimating the RBM.
        agent_vars : AgentVars
            A custom class/object containing RBM agent variables.
        df_for : pd.DataFrame
            Data frame containing input required for regression analysis.
        df_rbm : pd.DataFrame
            Data frame containing input required for RBM analysis.
        force_rerun : bool, optional
            Determines whether to force re-running of regression models, by default False.

        Returns
        -------
        None
            This function does not return any value.
        """

        # Initialize output lists
        self.all_quest_data = []
        self.all_parameters = []
        self.all_results = []

        # Initialize counter for significant results
        self.significance_counter = 0

        # Initialize counter for specifications
        self.counter = 0

        # Cycle over all specifications and compute correlation
        # -----------------------------------------------------

        # Regression specifications
        for reg_name, reg_spec in regression_specs.items():

            # Dependent variable (currently fixed or adaptive LR)
            for dep_name, dep_spec in reg_spec["dependent_variable"].items():

                if dep_spec:

                    # Compute or load regression result (if pre-computed)
                    df_reg = run_or_load_regression(
                        reg_name,
                        reg_spec[reg_name],
                        reg_vars,
                        df_for,
                        force_rerun=force_rerun,
                    )

                    # Here, we only need the group info
                    df_group = df_reg[["ID", "group"]]

                    # Flip beta_4 (adaptive LR) to ensure all effects go in same direction:
                    # Naturally, higher beta_4 is more adaptive, but beta_1 is the other way around.
                    # We flip beta_4 to ensure all effects are consistent.
                    if dep_name == "beta_4":
                        df_reg["beta_4"] *= -1

                    # Update counter
                    self.counter += 1

                    # Compute t-test
                    self.compute_sca_ttest(
                        df_group,
                        df_reg,
                        dep_name,
                        reg_spec[reg_name],
                    )

        # RBM specifications
        for rbm_name, rbm_spec in rbm_specs.items():

            # Compute or load regression result (if pre-computed)
            df_model = run_or_load_rbm(
                rbm_name,
                rbm_spec,
                est_vars,
                agent_vars,
                df_rbm,
                force_rerun=force_rerun,
            )

            # Here, we only need the group info
            df_group = df_model[["ID", "group"]]

            # We currently on have hazard rate for RBM
            which_var_rbm = "h"

            # Update counter
            self.counter += 1

            # Compute t-test
            self.compute_sca_ttest(df_group, df_model, which_var_rbm, rbm_spec)

        print(f"Number of significant results: {self.significance_counter}")

    def compute_sca_ttest(
        self,
        df_group: pd.DataFrame,
        df_model: pd.DataFrame,
        dep_name: str,
        reg_spec: dict,
    ) -> None:
        """Compute t-test for group comparison in the Vaghi data set.

        Parameters
        ----------
        df_group : pd.DataFrame
            DataFrame containing group information.
        df_model : pd.DataFrame
            DataFrame containing coefficients.
        dep_name : str
            Name of the dependent variable.
        reg_spec : dict
            Model specification.

        Returns
        -------
        None
            This function does not return any value.
        """

        # Perform t-test for group comparison
        res = scipy.stats.ttest_ind(
            df_model.loc[df_model["group"] == 0, dep_name],
            df_model.loc[df_model["group"] == 1, dep_name],
        )
        bic = df_model["BIC"].to_numpy()

        # Combine results in dictionary
        analysis_result = {
            "effect": res.statistic,
            "p_value": res.pvalue,
            "dv": dep_name,
            "bic": np.sum(bic),
        }

        # Count number of single-test significant results
        if analysis_result.get("p_value", 1.0) <= 0.05:
            self.significance_counter += 1

        # analysis_spec = reg_spec[reg_name]
        analysis_spec = reg_spec.copy()

        # Record all spec info
        flat_spec_result = {
            "model_id": f"model_{self.counter}",
            "dependent_variable": dep_name,
            **analysis_spec,
            **analysis_result,
        }

        # Save results: We need the spec info later for plotting the design choices
        # -------------------------------------------------------------------------

        # Get hash for full specification
        full_spec = {"analysis": analysis_spec, "which_var_analysis": dep_name}
        sca_hash = get_hash(full_spec)

        # Save full specification results
        spec_result = pd.DataFrame([flat_spec_result])
        spec_result.name = f"sca_{sca_hash}"
        safe_save_dataframe(
            spec_result, data_dir="for_data/sca_temp/", print_action=False
        )

        # Save group information and learning rates for permutation testing
        # -----------------------------------------------------------------

        # Add IDs to questionnaire variables of interest
        df_group = df_group.set_index("ID")[["group"]].astype(float).sort_index()

        # Add IDs to learning parameters of interest
        parameters_vector = (
            df_model[[dep_name, "ID"]].set_index("ID").astype(float).sort_index()
        )

        # Save for permutation test
        self.all_quest_data.append(df_group)
        self.all_parameters.append(parameters_vector)
        self.all_results.append(analysis_result)

    def compute_sca_correlations(
        self,
        df_questionnaires: pd.DataFrame,
        df_analysis: pd.DataFrame,
        which_var_analysis: str,
        which_var_quest: str,
        analysis_spec: dict,
    ) -> None:
        """Compute correlation between questionnaire scores and coefficients.

        Parameters
        ----------
        df_questionnaires : pd.DataFrame
            DataFrame containing questionnaire scores.
        df_analysis : pd.DataFrame
            DataFrame containing regression or RBM coefficients.
        which_var_analysis : str
            Variable of interest in the regression or RBM coefficients.
        which_var_quest : str
            Questionnaire variable(s) of interest
        analysis_spec : dict
            Analysis specification.

        Returns
        -------
        None
            This function does not return any value.
        """

        # Unify subject column name
        if "ID" in df_questionnaires.columns:
            pass
        elif "subj_num" in df_questionnaires.columns:
            df_questionnaires = df_questionnaires.rename(columns={"subj_num": "ID"})

        # Get IDs in both regression and questionnaire scores (where some filled out Qs incompletely)
        common_ids = set(df_analysis["ID"]) & set(df_questionnaires["ID"])

        # Filter data frames
        df_analysis = df_analysis[df_analysis["ID"].isin(common_ids)].reset_index(
            drop=True
        )
        df_questionnaires = df_questionnaires[
            df_questionnaires["ID"].isin(common_ids)
        ].reset_index(drop=True)

        # Select variable of interest for correlation analysis
        df_analysis_voi = df_analysis[which_var_analysis].to_numpy()
        df_questionnaires_voi = df_questionnaires[which_var_quest].to_numpy()
        bic = df_analysis["BIC"].to_numpy()

        # Sanity check if we have enough overlapping subjects
        assert (
            len(df_questionnaires_voi) == len(df_analysis_voi)
            and len(df_questionnaires_voi) == self.expected_n_subj
        ), "Not enough overlapping subjects"

        # Compute correlation
        r, p = stats.spearmanr(df_questionnaires_voi, df_analysis_voi)

        analysis_result = {
            "effect": r,
            "p_value": p,
            "dv": which_var_analysis,
            "qv": which_var_quest,
            "bic": np.sum(bic),
        }

        # Count number of single-test significant results
        if analysis_result.get("p_value", 1.0) <= 0.05:
            self.significance_counter += 1

        # Record all spec info
        flat_spec_result = {
            "model_id": f"model_{self.counter}",
            "dependent_variable": which_var_analysis,
            "quest_variable": which_var_quest,
            **analysis_spec,
            **analysis_result,
        }

        # Save results: We need the spec info later for plotting the design choices
        # -------------------------------------------------------------------------

        # Get hash for full specification
        full_spec = {
            "analysis": analysis_spec,
            "which_var_analysis": which_var_analysis,
            "which_var_quest": which_var_quest,
        }
        sca_hash = get_hash(full_spec)

        # Save full specification results
        spec_result = pd.DataFrame([flat_spec_result])
        spec_result.name = f"sca_{sca_hash}"
        safe_save_dataframe(
            spec_result, data_dir="for_data/sca_temp/", print_action=False
        )

        # Save questionnaire scores and learning rates for permutation testing
        # --------------------------------------------------------------------

        # Add IDs to questionnaire variables of interest
        questionnaire_vector = (
            df_questionnaires.set_index("ID")[[which_var_quest]]
            .astype(float)
            .sort_index()
        )

        # Add IDs to learning parameters of interest
        parameters_vector = (
            df_analysis[[which_var_analysis, "ID"]]
            .set_index("ID")
            .astype(float)
            .sort_index()
        )

        # Save for permutation test
        self.all_quest_data.append(questionnaire_vector)
        self.all_parameters.append(parameters_vector)
        self.all_results.append(analysis_result)

    def run_sca_fa(
        self,
        regression_specs: dict,
        analysis_specs: dict,
        reg_vars: RegVars,
        df_for: pd.DataFrame,
        which_factor: pd.DataFrame,
        which_var: str = "beta_1",
        force_rerun=False,
    ) -> tuple[list, list, list]:
        """Run specification curve analysis (SCA) for multiple regression and factor analysis specifications.

        Parameters
        ----------
        regression_specs : dict
            Dictionary containing regression specifications, where each key specifies
            a regression name and its corresponding value is a specification dictionary.
        analysis_specs : dict
            Dictionary containing factor analysis specifications. Each key specifies
            a factor analysis name and its configuration.
        reg_vars : RegVars
            A custom class/object containing variables and settings necessary for executing
            regression models.
        df_for : pd.DataFrame
            Data frame containing input required for regression analysis.
        which_factor : pd.DataFrame
            Data frame indicating which factor is used for the specifications.
        which_var : str, optional
            The target variable of interest in regression data, default is "beta_1".
        force_rerun : bool, optional
            Determines whether to force re-running of regression models, by default False.

        Returns
        -------
        tuple of lists
            A tuple containing:
            - all_factors : list of pd.Series
              List containing factor scores indexed by subject (for each factor analysis
              specification).
            - all_parameters : list of pd.DataFrame
              List of regression coefficients (e.g., learning rates), indexed by subject.
            - all_results : list of dict
              List containing analysis results (e.g., correlation coefficients and p-values)
              for each specification combination.
        """

        # Initialize output lists
        all_factors = []
        all_parameters = []
        all_results = []

        # Initialize counter for significant results
        self.significance_counter = 0

        # Initialize counter for specifications
        counter = 0

        # Cycle over all specifications and compute correlation
        # -----------------------------------------------------

        # Regression specifications
        for reg_name, reg_spec in regression_specs.items():

            # Compute or load regression result (if pre-computed)
            df_reg = run_or_load_regression(
                reg_name, reg_spec, reg_vars, df_for, force_rerun=force_rerun
            )

            # Factor analysis specifications
            for analysis_name, analysis_spec in analysis_specs.items():

                # Update counter
                counter += 1

                # Select matching subjects
                df_sca_fa, df_reg, fa_hash = self.filter_subjects(analysis_spec, df_reg)

                # Compute correlation between factor scores and regression coefficients
                analysis_result, df_sca_fa, df_reg, factor_name = self.correlate_reg_fa(
                    which_factor, fa_hash, df_sca_fa, df_reg, which_var
                )

                # Count number of single-test significant results
                if analysis_result.get("p_value", 1.0) <= 0.05:
                    self.significance_counter += 1

                # Record all spec info
                flat_spec_result = {
                    "model_id": f"model_{counter}",
                    **reg_spec,
                    **analysis_spec,
                    **analysis_result,
                }

                # Save results: We need the spec info later for plotting the design choices
                # -------------------------------------------------------------------------

                # Get hash for full specification (regression + factor analysis)
                full_spec = {"regression": reg_spec, "analysis": analysis_spec}
                sca_hash = get_hash(full_spec)

                # Save full specification results
                spec_result = pd.DataFrame([flat_spec_result])
                spec_result.name = f"sca_{sca_hash}"
                safe_save_dataframe(
                    spec_result, data_dir="for_data/sca_temp/", print_action=False
                )

                # Save factor scores and learning rates for permutation testing
                # -------------------------------------------------------------

                # Add ID to factor scores
                factor_series = (
                    df_sca_fa.set_index("ID")[factor_name].astype(float).sort_index()
                )

                # Add ID to learning rates
                betas_vector = (
                    df_reg[[which_var, "ID"]].set_index("ID").astype(float).sort_index()
                )

                # Ensure that both cover the same IDs
                if not factor_series.index.equals(betas_vector.index):
                    missing_x = betas_vector.index.difference(factor_series.index)
                    missing_y = factor_series.index.difference(betas_vector.index)
                    raise ValueError(
                        f"Index mismatch between FA and betas: "
                        f"{len(missing_x)} extra in betas, {len(missing_y)} extra in FA."
                    )

                # Save for permutation test
                all_factors.append(factor_series)
                all_parameters.append(betas_vector)
                all_results.append(analysis_result)

        print(f"Number of significant results: {self.significance_counter}")

        return all_factors, all_parameters, all_results

    def filter_subjects(
        self, analysis_spec: dict, df_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """Filter subjects based on common IDs between the factor scores and the data frame of interest
        (regression results, questionnaire scores, etc.).

        Parameters
        ----------
        analysis_spec : dict
            Current factor analysis specification.
        df_data : pd.DataFrame
            Other data frame of interest (e.g., regression results, questionnaire scores, etc.).

        Returns
        -------
        pd.DataFrame
            Data frame containing factor scores; matched with regression subjects.
        pd.DataFrame
            Other data frame of interest; matched with FA subjects.
        str
            Hash code for the current factor analysis specification.
        """

        # Get hash code for current FA specification and load FA scores
        test_FA_spec = {k: analysis_spec[k] for k in sorted(analysis_spec.keys())}
        fa_hash = get_hash(test_FA_spec["fa"])

        curr_file_name = (
            self.sca_fa_folder + "/sca_fa_" + fa_hash + self.sca_fa_filetype
        )

        if self.sca_fa_filetype == ".csv":
            df_sca_fa = pd.read_csv(curr_file_name)
        else:
            df_sca_fa = pd.read_pickle(curr_file_name)

        # Unify subject column name
        if "ID" in df_sca_fa.columns:
            pass
        elif "subj_num" in df_sca_fa.columns:
            df_sca_fa = df_sca_fa.rename(columns={"subj_num": "ID"})
        # futuretodo: fix in fa file
        elif "V1" in df_sca_fa.columns:
            df_sca_fa = df_sca_fa.rename(columns={"V1": "ID"})
        else:
            raise RuntimeError("FA table must contain 'ID' or 'subj_num'.")

        # Sort by subject ID and reset index
        df_sca_fa = df_sca_fa.sort_values(by=["ID"]).reset_index(drop=True)

        # Get IDs in both data frames (where some filled out Qs incompletely)
        common_ids = set(df_data["ID"]) & set(df_sca_fa["ID"])

        # Filter data frames based on common IDs
        df_data = df_data[df_data["ID"].isin(common_ids)].reset_index(drop=True)

        # Filter questionnaire data based on common IDs
        df_sca_fa = df_sca_fa[df_sca_fa["ID"].isin(common_ids)].reset_index(drop=True)

        return df_sca_fa, df_data, fa_hash

    def correlate_reg_fa(
        self,
        which_factor: pd.DataFrame,
        fa_hash: str,
        df_sca_fa: pd.DataFrame,
        df_reg: pd.DataFrame,
        which_var: str,
    ) -> tuple[dict, pd.DataFrame, pd.DataFrame, str]:
        """Correlate regression model results with factor scores.

        The function matches IDs of subjects in the regression model results and the factor scores.

        Parameters
        ----------
        which_factor : pd.DataFrame
            Factor scores and indices.
        fa_hash : str
            Hash code for the current factor analysis specification.
        df_sca_fa : pd.DataFrame
            Filtered data frame containing factor scores for the SCA.
        df_reg : pd.DataFrame
            Filtered regression model results.
        which_var : str
            Regression variable of interest.

        Returns
        -------
        dict
            Analysis results (correlation, p-value).
        pd.DataFrame
            Current regression model results
        pd.DataFrame
            Data frame containing factor scores
        str
            Factor score of interest.
        """

        # We are using the factor with the highest correlation with CAPE
        factor_name = which_factor[which_factor["fa_hash"] == fa_hash][
            "max_index"
        ].iloc[0]

        # Select variable of interest for correlation analysis
        df_fa_voi = df_sca_fa[factor_name].to_numpy()
        df_reg_voi = df_reg[which_var].to_numpy()

        # Sanity check if we have enough overlapping subjects
        assert (
            len(df_fa_voi) == len(df_reg_voi) and len(df_fa_voi) == self.expected_n_subj
        ), "Not enough overlapping subjects"

        # Compute correlation
        r, p = stats.spearmanr(df_fa_voi, df_reg_voi)

        analysis_result = {
            "effect": r,
            "p_value": p,
        }

        return analysis_result, df_sca_fa, df_reg, factor_name

    def run_permutation_test(self, n_perm: int = 1000) -> float:
        """Run permutation test for specification curve analysis.

        Parameters
        ----------
        n_perm : int, optional
            Number of permutations, by default 1000.

        Returns
        -------
        float
            Permutation-test-based p-value.
        """

        # Extract IDs from factor scores
        common_ids = set(self.all_quest_data[0].index)
        common_ids = sorted(common_ids)

        # Sanity check if we have enough overlapping subjects
        assert (
            len(common_ids) == self.expected_n_subj
        ), "Not enough overlapping subjects"

        # Put factors and betas into matrices for permutation testing
        all_quest_data_matrix = np.column_stack(
            [x.reindex(common_ids).to_numpy() for x in self.all_quest_data]
        )
        all_parameters_matrix = np.column_stack(
            [y.reindex(common_ids).to_numpy() for y in self.all_parameters]
        )

        # Compute correlation between factors and betas (not permuted)
        rs_obs = []
        for j, (col_q, col_p) in enumerate(
            zip(all_quest_data_matrix.T, all_parameters_matrix.T)
        ):

            r, _ = stats.spearmanr(col_q, col_p)
            rs_obs.append(r)

            # Check whether r-values are consistent with original results
            r_prev = self.all_results[j]["effect"]
            npt.assert_almost_equal(
                r, r_prev, decimal=7, err_msg="Correlation mismatch"
            )

        # Compute Fisher z-score as test statistic
        z_value_obs = fisher_z_median(rs_obs)

        # Permutation test: Same subject permutation applied to all columns of all_parameters_matrix
        rng = default_rng(42)
        z_values_perm = np.empty(n_perm, dtype=np.float32)

        # Inform user
        sleep(0.1)
        print("\nRunning permutation test:")
        sleep(0.1)

        # Initialize progress bar
        pbar = tqdm(total=n_perm)

        # Cycle over permutations
        for i in range(n_perm):

            # Permute subject row
            perm = rng.permutation(all_quest_data_matrix.shape[0])

            # Apply to every spec simultaneously
            all_quest_data_perm = all_quest_data_matrix[perm, :]

            # Initialize list with correlations
            r_list = []

            # Cycle over factor analysis permutations
            # futuretodo: consider freedman-lane when using regression and not only correlation
            for j, (col_q, col_p) in enumerate(
                zip(all_quest_data_perm.T, all_parameters_matrix.T)
            ):
                r, _ = stats.spearmanr(col_q, col_p)
                r_list.append(r)

            # Compute Fisher z-score of permuted correlations
            z_values_perm[i] = fisher_z_median(r_list)

            # Update progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()

        # Compute p-value based on Fisher z-score
        p_value = perm_pval(z_value_obs, z_values_perm, tail="two-sided")
        print("\nPermutation p-value (median-based):", p_value)

        # Plot histogram of permutation test results
        plt.figure()
        plt.hist(z_values_perm, bins=50)
        plt.axvline(z_value_obs, color="red", linestyle="dashed")
        title = (
            "Observed z-value = "
            + str(np.round(z_value_obs, 3))
            + " with p = "
            + str(np.round(p_value, 3))
        )
        plt.title(title)
        sns.despine()
        plt.savefig("figures/p_value_hist_" + self.which_analysis_str + ".png", dpi=300)

        return p_value

    def run_permutation_test_vaghi(self, n_perm: int = 1000) -> float:
        """Run permutation test for the Vaghi specification curve analysis.

        Parameters
        ----------
        n_perm : int, optional
            Number of permutations, by default 1000.

        Returns
        -------
        float
            Permutation-test-based p-value.
        """

        # Extract IDs from factor scores
        common_ids = set(self.all_quest_data[0].index)
        common_ids = sorted(common_ids)

        # Sanity check if we have enough overlapping subjects
        assert (
            len(common_ids) == self.expected_n_subj
        ), "Not enough overlapping subjects"

        # Put group data and betas into matrices for permutation testing
        all_quest_data_matrix = np.column_stack(
            [x.reindex(common_ids).to_numpy() for x in self.all_quest_data]
        )
        all_parameters_matrix = np.column_stack(
            [y.reindex(common_ids).to_numpy() for y in self.all_parameters]
        )

        # Assertion checks to ensure consistent ordering
        # -----------------------------------------------

        # Check that all DataFrames have the same subjects in the same order after reindexing
        for i, quest_df in enumerate(self.all_quest_data):
            reindexed = quest_df.reindex(common_ids)
            assert (
                reindexed.index.tolist() == common_ids
            ), f"Questionnaire data {i}: Index mismatch after reindexing"

        for i, param_df in enumerate(self.all_parameters):
            reindexed = param_df.reindex(common_ids)
            assert (
                reindexed.index.tolist() == common_ids
            ), f"Parameter data {i}: Index mismatch after reindexing"

        # Check matrix shapes match
        assert (
            all_quest_data_matrix.shape[0] == all_parameters_matrix.shape[0]
        ), f"Row count mismatch: quest={all_quest_data_matrix.shape[0]}, params={all_parameters_matrix.shape[0]}"

        assert all_quest_data_matrix.shape[0] == len(
            common_ids
        ), f"Matrix rows ({all_quest_data_matrix.shape[0]}) don't match common_ids length ({len(common_ids)})"

        # Check that the number of columns matches the number of specifications
        assert all_quest_data_matrix.shape[1] == len(
            self.all_quest_data
        ), f"Quest matrix columns ({all_quest_data_matrix.shape[1]}) don't match list length ({len(self.all_quest_data)})"

        assert all_parameters_matrix.shape[1] == len(
            self.all_parameters
        ), f"Param matrix columns ({all_parameters_matrix.shape[1]}) don't match list length ({len(self.all_parameters)})"

        assert len(self.all_quest_data) == len(
            self.all_parameters
        ), f"Length mismatch: quest={len(self.all_quest_data)}, params={len(self.all_parameters)}"

        # Compute correlation between factors and betas (not permuted)
        ts_obs = []

        # Cycle over subjects
        for j, (col_q, col_p) in enumerate(
            zip(all_quest_data_matrix.T, all_parameters_matrix.T)
        ):
            # Perform t-test comparing the groups
            res = scipy.stats.ttest_ind(col_p[col_q == 0], col_p[col_q == 1])
            t_value = res.statistic

            # Check whether t-values are consistent with original results
            t_prev = self.all_results[j]["effect"]
            npt.assert_almost_equal(
                t_value, t_prev, decimal=7, err_msg="T-value mismatch"
            )
            ts_obs.append(t_value)

        # todo: perform permutation test with betas, not t-values
        t_value_obs = float(np.median(ts_obs))

        # Permutation test: Same subject permutation applied to all columns of all_parameters_matrix
        rng = default_rng(42)
        t_values_perm = np.empty(n_perm, dtype=np.float32)

        # Inform user
        sleep(0.1)
        print("\nRunning permutation test:")
        sleep(0.1)

        # Initialize progress bar
        pbar = tqdm(total=n_perm)

        # Cycle over permutations
        for i in range(n_perm):

            # Permute subject row
            perm = rng.permutation(all_quest_data_matrix.shape[0])

            # Apply to every spec simultaneously
            all_quest_data_perm = all_quest_data_matrix[perm, :]

            # Initialize list with correlations
            t_list = []

            # Cycle over factor analysis permutations
            for j, (col_q, col_p) in enumerate(
                zip(all_quest_data_perm.T, all_parameters_matrix.T)
            ):
                res = scipy.stats.ttest_ind(col_p[col_q == 0], col_p[col_q == 1])
                t_value = res.statistic
                t_list.append(t_value)

            # Median t-value
            t_values_perm[i] = float(np.median(t_list))

            # Update progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()

        # Compute p-value based on t-values
        p_value = perm_pval(t_value_obs, t_values_perm, tail="two-sided")
        print("\nPermutation p-value (median-based):", p_value)

        # Plot histogram of permutation test results
        plt.figure()
        plt.hist(t_values_perm, bins=50)
        plt.axvline(t_value_obs, color="red", linestyle="dashed")
        title = (
            "Observed t-value = "
            + str(np.round(t_value_obs, 3))
            + " with p = "
            + str(np.round(p_value, 3))
        )
        plt.title(title)
        sns.despine()
        plt.savefig("figures/p_value_hist_" + self.which_analysis_str + ".png", dpi=300)

        return p_value

    def fast_model_comp(
        self, all_voi: np.ndarray, all_bic: np.ndarray, voi_name: str = "Learning Rate"
    ) -> None:
        """Runs a quick-and-dirty model comparison for SCA.

        Parameters
        ----------
        all_voi : np.ndarray
            All variable of interests (e.g., learning rate) for single-subject plots.
        all_bic : np.ndarray
            All BICs for single-subject and sum plots.
        voi_name : str
            Y-axis string for variable of interest.

        Returns
        -------
        None
            This function does not return any value.
        """

        # Convert to long format for stripplot
        bic_df = pd.DataFrame(
            all_bic, columns=[f"Model_{i}" for i in range(np.size(all_bic, 1))]
        )
        bic_long = bic_df.melt(var_name="model", value_name="BIC")
        bic_long["model_idx"] = bic_long["model"].str.extract("(\d+)").astype(int)

        plt.figure()
        ax = plt.gca()
        sns.barplot(
            data=bic_long,
            x="model_idx",
            y="BIC",
            color="k",
            alpha=0.7,
            errorbar=None,
            ax=ax,
        )
        sns.stripplot(
            data=bic_long, x="model_idx", y="BIC", color="k", alpha=0.7, size=2, ax=ax
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Bayesian Information Criterion")
        sns.despine()
        plt.savefig(
            "figures/sca_BICs_mean_" + self.which_analysis_str + ".png", dpi=300
        )

        # Plot sum of BIC across all subjects for model comparison
        bic_sum = np.sum(all_bic, axis=0)

        # Plot BIC for quick and dirty model comparison
        plt.figure()
        plt.bar(np.arange(len(bic_sum)), bic_sum)
        ax = plt.gca()
        ax.set_xlabel("Model")
        ax.set_ylabel("Bayesian Information Criterion")
        sns.despine()
        plt.savefig(
            "figures/sca_BICs_sum_" + self.which_analysis_str + "_BIC.png", dpi=300
        )

        # Convert to long format for stripplot
        voi_df = pd.DataFrame(
            all_voi, columns=[f"Model_{i}" for i in range(np.size(all_voi, 1))]
        )
        voi_long = voi_df.melt(var_name="model", value_name="voi")
        voi_long["model_idx"] = voi_long["model"].str.extract("(\d+)").astype(int)

        plt.figure()
        ax = plt.gca()
        sns.barplot(
            data=voi_long,
            x="model_idx",
            y="voi",
            color="k",
            alpha=0.7,
            errorbar=None,
            ax=ax,
        )
        sns.stripplot(
            data=voi_long, x="model_idx", y="voi", color="k", alpha=0.7, size=2, ax=ax
        )
        ax.set_xlabel("Model")
        ax.set_ylabel(voi_name)
        sns.despine()
        plt.savefig("figures/sca_vois_" + self.which_analysis_str + ".png", dpi=300)

    def plot_sca(
        self,
        p_T1: float,
        sca_path: str = "for_data/sca_temp/",
        ylabel: str = "Effect Size",
    ) -> None:
        """Plot the specification curve analysis results.

        Parameters
        ----------
        p_T1 : float
            Permutation-based p-value.
        ylabel : str, optional
            Y-axis label, by default "Effect Size".

        Returns
        -------
        None
            This function does not return any value.
        """

        # ------------
        # Get SCA data
        # ------------

        # Get all SCA files
        result_files = glob.glob(sca_path + "*.pkl")

        # Initialize list summarizing all specifications
        summary_rows = []

        # Cycle over all SCAs to extract exact specifications
        for file in result_files:
            # Load current file and add to summary list
            result = pd.read_pickle(file)
            row = result.iloc[0]
            summary_rows.append(row.to_dict())

        # Convert list to DataFrame
        summary_df = pd.DataFrame(summary_rows)

        # Sort specifications by effect size
        plot_df = summary_df.copy()
        plot_df = plot_df.sort_values("effect").reset_index(drop=True)

        # ----------------------------
        # Create binary design matrix
        # ----------------------------

        # Regression design matrix
        # ------------------------

        # Initialize rows
        n_specs = len(plot_df)
        cape = np.full(n_specs, np.nan)
        spq = np.full(n_specs, np.nan)
        beta_2 = np.full(n_specs, np.nan)
        beta_3 = np.full(n_specs, np.nan)
        beta_4 = np.full(n_specs, np.nan)
        beta_5 = np.full(n_specs, np.nan)
        beta_6 = np.full(n_specs, np.nan)
        beta_7 = np.full(n_specs, np.nan)
        surprise = np.full(n_specs, np.nan)
        uncertainty = np.full(n_specs, np.nan)
        catch_trial = np.full(n_specs, np.nan)
        blank_line = np.full(n_specs, np.nan)
        hazard_rate = np.full(n_specs, np.nan)
        fixed_lr = np.full(n_specs, np.nan)
        adaptive_lr = np.full(n_specs, np.nan)
        pers_reg = np.full(n_specs, np.nan)
        pers_rbm = np.full(n_specs, np.nan)

        # Cycle over all specifications
        for i in range(n_specs):

            if "quest_variable" in plot_df.columns:
                if plot_df.loc[i, "quest_variable"] == "SPQ1":
                    spq[i] = True
                else:
                    spq[i] = False

                if plot_df.loc[i, "quest_variable"] == "SPQ1":
                    cape[i] = False
                else:
                    cape[i] = True

            if plot_df.loc[i, "dependent_variable"] == "h":
                hazard_rate[i] = True
            else:
                hazard_rate[i] = False

            if plot_df.loc[i, "dependent_variable"] == "beta_1":
                fixed_lr[i] = True
            else:
                fixed_lr[i] = False

            if plot_df.loc[i, "dependent_variable"] == "beta_4":
                adaptive_lr[i] = True
            else:
                adaptive_lr[i] = False

            if (
                plot_df.loc[i, "lambda_0"] == True
                and plot_df.loc[i, "dependent_variable"] == "h"
            ):
                pers_rbm[i] = True
            else:
                pers_rbm[i] = False

            if (
                plot_df.loc[i, "lambda_0"] == True
                and not plot_df.loc[i, "dependent_variable"] == "h"
            ):
                pers_reg[i] = True
            else:
                pers_reg[i] = False

            # Extract regression coefficients of interest
            beta_2[i] = plot_df.loc[i, "beta_2"]
            beta_3[i] = plot_df.loc[i, "beta_3"]
            beta_4[i] = plot_df.loc[i, "beta_4"]
            beta_5[i] = plot_df.loc[i, "beta_5"]
            beta_6[i] = plot_df.loc[i, "beta_6"]
            beta_7[i] = plot_df.loc[i, "beta_7"]

            surprise[i] = plot_df.loc[i, "s"]
            uncertainty[i] = plot_df.loc[i, "u"]
            catch_trial[i] = plot_df.loc[i, "sigma_H"]

        # Summarize combinations
        # ----------------------

        # Combine into one data frame
        reg_onehot = pd.DataFrame(
            data={
                "Regression": blank_line,
                "CPP": beta_2,
                "RU": beta_3,
                "Learning rate": beta_4,
                "Reward": beta_5,
                "Variability": beta_6,
                "Perseveration (regression)": pers_reg,
                "Catch trial (regression)": beta_7,
                "Fixed learning rate (dv)": fixed_lr,
                "Adaptive learning rate (dv)": adaptive_lr,
                "": blank_line,
                "RBM estimation": blank_line,
                "Surprise": surprise,
                "Uncertainty": uncertainty,
                "Perseveration (RBM)": pers_rbm,
                "Catch trial (RBM)": catch_trial,
                "Hazard rate (dv)": hazard_rate,
            }
        )

        if "quest_variable" in plot_df.columns:

            if plot_df["quest_variable"].isin(["SPQ1"]).any():

                # Combine into one data frame
                reg_onehot_quest_addon = pd.DataFrame(
                    data={
                        " ": blank_line,
                        "Questionnaire": blank_line,
                        "SPQ": spq,
                        "CAPE": cape,
                    }
                )

                reg_onehot = pd.concat([reg_onehot, reg_onehot_quest_addon], axis=1)

        if np.sum(reg_onehot["Variability"]) == 0:
            reg_onehot = reg_onehot.drop(columns=["Variability"])

        if np.sum(reg_onehot["Catch trial (regression)"]) == 0:
            reg_onehot = reg_onehot.drop(columns=["Catch trial (regression)"])

        if np.sum(reg_onehot["Catch trial (RBM)"]) == 0:
            reg_onehot = reg_onehot.drop(columns=["Catch trial (RBM)"])

        if np.sum(reg_onehot["Perseveration (regression)"]) == 0:
            reg_onehot = reg_onehot.drop(columns=["Perseveration (regression)"])

        if np.sum(reg_onehot["Perseveration (RBM)"]) == 0:
            reg_onehot = reg_onehot.drop(columns=["Perseveration (RBM)"])

        # Combine all of the above into one spec matrix
        # ---------------------------------------------

        design_bin = pd.concat([reg_onehot], axis=1)  # flags,
        spec_matrix = (design_bin.T > 0).astype("uint8")
        spec_matrix.columns = plot_df["model_id"].astype(str)

        # Create the two subplots
        fig, (ax_main, ax_tile, ax_extra) = plt.subplots(
            3,
            1,
            figsize=(cm2inch(15, 18)),
            gridspec_kw={"height_ratios": [2, 2, 2]},
            sharex=True,
        )

        # Determine x-axis positions
        x_positions = np.arange(len(plot_df))
        x_shifted = x_positions + 0.5
        step = max(1, len(plot_df) // 10)

        # Plot the specification curve
        ax_main.plot(x_shifted, plot_df["effect"].to_numpy(), "-", color="k")
        ax_main.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax_main.set_ylabel(ylabel)
        ax_main.set_xticks(x_shifted[::step])
        ax_main.set_xticklabels(
            plot_df["model_id"].astype(str).tolist()[::step], rotation=90
        )
        ax_main.set_title(f"$p$ = " + str(round(p_T1, 3)))

        # Mark individual significant results
        significant_mask = plot_df["p_value"] <= 0.05
        ax_main.plot(
            x_shifted[significant_mask],
            plot_df["effect"][significant_mask].to_numpy(),
            ".",
            color="k",
        )

        # Color coding for different models
        for i, model_id in enumerate(spec_matrix):
            if spec_matrix.loc["Hazard rate (dv)", model_id] == 1:
                col_name = plot_df["model_id"].astype(str).iloc[i]
                mask = spec_matrix.loc[:, col_name] == 1
                spec_matrix.loc[mask, col_name] = 2
            elif spec_matrix.loc["Adaptive learning rate (dv)", model_id] == 1:
                col_name = plot_df["model_id"].astype(str).iloc[i]
                mask = spec_matrix.loc[:, col_name] == 1
                spec_matrix.loc[mask, col_name] = 3

        # Create colormap
        custom_cmap = ListedColormap(["#FFFFFF", "#89CFF0", "#088F8F", "#7393B3"])
        sns.heatmap(
            spec_matrix,
            ax=ax_tile,
            cbar=False,
            cmap=custom_cmap,
            vmin=0,
            vmax=3,
            linewidths=0,
        )
        ax_tile.set_ylabel("")
        ax_tile.set_xlabel("Specification")

        # Ensure all y-axis labels are shown
        ax_tile.set_yticks(np.arange(len(spec_matrix)) + 0.5)
        ax_tile.set_yticklabels(spec_matrix.index, rotation=0, fontsize=8)

        # Remove tick markers but keep labels
        ax_tile.tick_params(axis="y", which="both", left=False)

        # Print analysis types in bold font
        for label in ax_tile.get_yticklabels():
            if (
                label.get_text() == "Regression"
                or label.get_text() == "RBM estimation"
                or label.get_text() == "Questionnaire"
            ):
                label.set_weight("bold")

        ax_tile.set_xticks(x_shifted[::step])
        ax_tile.set_xticklabels(x_shifted[::step].astype(int), rotation=0)

        ax_extra.bar(x_shifted, plot_df["bic"].to_numpy(), color="k")
        ax_extra.set_ylabel("BIC (higher better)")
        sns.despine()
        plt.tight_layout()

        # Save figure
        output_file = "figures/sca_" + self.which_analysis_str + ".png"
        plt.savefig(output_file, dpi=400)

    # ----------------------------------------
    # Class methods related to factor analysis
    # ----------------------------------------

    def plot_sca_fa(
        self,
        p_T1: float,
        sca_path: str = "for_data/sca_temp/",
        ylabel: str = "Effect Size Fixed Learning Rate",
    ) -> None:
        """Plot the specification curve analysis results.

        Parameters
        ----------
        p_T1 : float
            Permutation-based p-value.
        ylabel : str, optional
            Y-axis label, by default "Effect Size Fixed Learning Rate".

        Returns
        -------
        None
            This function does not return any value.
        """

        # ------------
        # Get SCA data
        # ------------

        # Get all SCA files
        result_files = glob.glob(sca_path + "*.pkl")

        # Initialize list summarizing all specifications
        summary_rows = []

        # Cycle over all SCAs to extract exact specifications
        for file in result_files:
            # Load current file and add to summary list
            result = pd.read_pickle(file)
            row = result.iloc[0]
            summary_rows.append(row.to_dict())

        # Convert list to DataFrame
        summary_df = pd.DataFrame(summary_rows)

        # Sort specifications by effect size
        plot_df = summary_df.copy()
        plot_df = plot_df.sort_values("effect").reset_index(drop=True)

        # ----------------------------
        # Create binary design matrix
        # ----------------------------

        # Regression design matrix
        # ------------------------

        # Initialize rows
        n_specs = len(plot_df)  # number of specifications
        norm_sep = np.full(n_specs, np.nan)  # normative terms separated
        norm_comb = np.full(n_specs, np.nan)  # normative terms combined
        covariates = np.full(n_specs, np.nan)  # covariates included in model

        # Cycle over all specifications
        for i in range(n_specs):

            # Extract regression coefficients of interest
            beta_2 = plot_df.loc[i, "beta_2"]
            beta_3 = plot_df.loc[i, "beta_3"]
            beta_4 = plot_df.loc[i, "beta_4"]
            beta_5 = plot_df.loc[i, "beta_5"]
            beta_6 = plot_df.loc[i, "beta_6"]
            beta_7 = plot_df.loc[i, "beta_7"]

            # Summarize what combinations mean
            # --------------------------------

            # Normative terms separated
            if beta_2 == 1 or beta_3 == 1:
                norm_sep[i] = True
            else:
                norm_sep[i] = False

            # Normative terms combined
            if beta_4 == 1:
                norm_comb[i] = True
            else:
                norm_comb[i] = False

            # Covariates included in model
            if beta_5 == 1 or beta_6 == 1 or beta_7 == 1:
                covariates[i] = True
            else:
                covariates[i] = False

        # Combine into one data frame
        reg_onehot = pd.DataFrame(
            data={
                "norm_sep": norm_sep,
                "norm_comb": norm_comb,
                "covariates": covariates,
            }
        )

        # Factor analysis design matrix
        # -----------------------------

        # Extract FA metadata directly from the fa column
        fa_variables = [
            "analysis_type",
            "rotation",
            "factor_method",
            "fs_method",
            "n_factors",
        ]
        fa_meta = pd.DataFrame(plot_df["fa"].tolist())[fa_variables]
        fa_onehot = pd.get_dummies(
            fa_meta[["analysis_type", "rotation", "factor_method", "fs_method"]].fillna(
                "NA"
            ),
            prefix=["type", "rot", "meth", "fs"],
            dtype=int,
        )

        # Do the same for number of factors
        fa_nf = pd.get_dummies(fa_meta["n_factors"], prefix="nfac", dtype=int)

        # Combine all of the above into one spec matrix
        # ---------------------------------------------

        design_bin = pd.concat([fa_nf, fa_onehot, reg_onehot], axis=1)  # flags,
        spec_matrix = (design_bin.T > 0).astype("uint8")
        spec_matrix.columns = plot_df["model_id"].astype(str)

        # Create the two subplots
        fig, (ax_main, ax_tile) = plt.subplots(
            2,
            1,
            figsize=(cm2inch(20, 15)),
            gridspec_kw={"height_ratios": [2, 2]},
            sharex=True,
        )

        # Determine x-axis positions
        x_positions = np.arange(len(plot_df))
        x_shifted = x_positions + 0.5
        step = max(1, len(plot_df) // 10)

        # Plot the specification curve
        ax_main.plot(x_shifted, plot_df["effect"].to_numpy(), "-")
        ax_main.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax_main.set_ylabel(ylabel)
        ax_main.set_xticks(x_shifted[::step])
        ax_main.set_xticklabels(
            plot_df["model_id"].astype(str).tolist()[::step], rotation=90
        )
        ax_main.set_title("p = " + str(round(p_T1, 3)))

        # Mark individual significant results
        significant_mask = plot_df["p_value"] <= 0.05
        ax_main.plot(
            x_shifted[significant_mask],
            plot_df["effect"][significant_mask].to_numpy(),
            color="red",
            linestyle="-",
        )
        # # Heatmap
        # sns.heatmap(
        #     spec_matrix.astype(float),
        #     ax=ax_tile,
        #     cbar=False,
        #     cmap=bw,
        #     vmin=0,
        #     vmax=1,
        #     linewidths=0,
        # )
        # ax_tile.set_ylabel("Design Choice")
        # ax_tile.set_xlabel("")

        # Heatmap
        # Create a custom matrix
        significant_mask = plot_df["p_value"] <= 0.05
        custom_matrix = spec_matrix.astype(float).copy()

        # Set significant columns' black cells (value 1) to red (value 2)
        for i, is_sig in enumerate(significant_mask):
            if is_sig:
                col_name = plot_df["model_id"].astype(str).iloc[i]
                # Only change cells that are 1 (black) to 2 (red)
                mask = custom_matrix.loc[:, col_name] == 1
                custom_matrix.loc[mask, col_name] = 2

        # Create colormap: white (0), black (1), red (2)
        custom_cmap = ListedColormap(["#FFFFFF", "#000000", "#FF0000"])
        sns.heatmap(
            custom_matrix,
            ax=ax_tile,
            cbar=False,
            cmap=custom_cmap,
            vmin=0,
            vmax=2,
            linewidths=0,
        )
        ax_tile.set_ylabel("Design Choice")
        ax_tile.set_xlabel("")

        # Ensure all y-axis labels are shown
        ax_tile.set_yticks(np.arange(len(spec_matrix)) + 0.5)
        ax_tile.set_yticklabels(spec_matrix.index, rotation=0, fontsize=8)

        ax_tile.set_xticks(x_shifted[::step])
        ax_tile.set_xticklabels(x_shifted[::step].astype(int), rotation=45)

        sns.despine()
        plt.tight_layout()

        # Save figure
        output_file = "figures/sca_" + self.which_analysis_str + ".png"
        plt.savefig(output_file, dpi=400)

    def check_fa_pool(self, all_expected=False) -> None:
        """Checks the validity and presence of required FA files in a specified folder.

        This function performs two checks:
        1. Verifies that all expected FA files based on the 'pool' are present in the
           specified folder.
        2. Optionally verifies that all files in the folder match expected hash codes
           when `all_expected` is set to True.

        Parameters
        ----------

        all_expected : bool, default=False
            If True, verifies that all files in the specified folder correspond to
            expected hash codes based on the 'pool'.

        Returns
        -------
        None
            This function does not return any value.
        """

        # Initialize hash list
        hash_list = []

        # 1. Check if our expected files exist in R folder
        # ------------------------------------------------

        folder = []

        # Loop over all FAs and compute hash codes
        for fa in self.pool:

            # Get hash code
            fa_hash = get_hash(fa)
            hash_list.append(fa_hash)

            # Check if file exists in data folder
            folder = Path(self.sca_fa_folder)
            found = any(folder.glob("sca_fa_" + fa_hash + self.sca_fa_filetype))

            # Stop when file is not found
            if not found:
                raise FileNotFoundError(f"File not found: {fa_hash}")

        # 2. Check if all R files match our files
        # ---------------------------------------

        if all_expected:

            # Set of our hash codes
            hash_set = {h for h in hash_list}

            # R FA json files
            files = list(folder.glob("*.pkl"))

            # Define regular expression
            tok_re = re.compile(r"[A-Za-z0-9]+")

            # Cycle over all R files
            for file in files:

                # Extract tokens from file name
                tokens = set(tok_re.findall(file.stem))

                # Check if tokens are in hash set
                hit = tokens & hash_set

                # Stop when file is not found
                if not hit:
                    raise FileNotFoundError(f"File not found: {file}")

    def build_specs_with_vars(self, spec_prefix="spec") -> dict:
        """This function builds factor-analysis combinations based on the constraints.

        Parameters
        ----------
        spec_prefix : str
            Prefix for the specification names.

        Returns
        -------
        dict
            Dictionary of specifications.
        """

        specs = {}
        per_flag = {f: 0 for f in self.var_names}

        # Cycle over factor analyses
        for fa in self.pool:

            # Extract hash code
            fa_hash = get_hash(fa)

            # Cycle over variables
            for var in self.var_names:

                # Skip if variable does not pass the rule
                if not self.passes_variable_rules(fa, var):
                    continue

                # Add to set of specifications
                spec = {f: (f == var) for f in self.var_names}
                spec["fa"] = deepcopy(fa)
                specs[f"{spec_prefix}_{var}_{fa_hash}"] = spec
                per_flag[var] += 1

        return specs

    def passes_variable_rules(self, fa_spec: dict, variable_name: str) -> bool:
        """Determines whether the variable rules are satisfied for a given factor-analysis
        specification and variable name.

        Parameters
        ----------
        fa_spec : dict
            A dictionary representing the factor-analysis specification.
        variable_name : str
            The name of the variable for which rules need to be validated.

        Returns
        -------
        bool
            True if all rules for the variable are satisfied; False otherwise.
        """

        # Get rules for this variable (empty list if variable not found)
        rules_for_variable = self.var_rules.get(variable_name, [])

        # Check each rule
        for rule_function in rules_for_variable:

            # Check for current factor-analysis specification if the specified rule applies
            if not rule_function(fa_spec):
                return False

        # Return True if all rules have been satisfied
        return True


# ----------------
# Global functions
# ----------------


def get_hash(spec: dict) -> str:
    """This function calculates the hash of a factor-analysis specification.

    Parameters
    ----------
    spec : dict
        Current specification.

    Returns
    -------
    str
        Hash code of the specification.
    """

    # Extract string of the specification
    s = json.dumps(spec, sort_keys=True, separators=(",", ":"))

    return hashlib.md5(s.encode()).hexdigest()


def run_or_load_regression(
    name: str,
    spec_dict: dict,
    reg_vars: "RegVars",
    df_for: pd.DataFrame,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """Runs a regression model or loads a saved model if it exists and meets the specified requirements.

    Parameters
    ----------
    name : str
        The name of the regression model, used for identifying saved files.
    spec_dict : dict
        A dictionary specifying the regression parameters and configurations.
    reg_vars : RegVars
        A custom object that holds regression variables and parameters required for model computation.
    df_for : pandas.DataFrame
        Data frame containing the input data used to run the model.
    force_rerun : bool, optional
        If True, forces re-computation of the model even if a saved version exists, by default False.

    Returns
    -------
    pd.DataFrame
        Regression data frame containing the results of the model.
    """

    # Get model hash
    reg_hash = get_hash(spec_dict)

    # Create file name
    filename = os.path.join(f"{name}_{reg_vars.n_sp}sp_{reg_hash}")
    path = os.path.join("for_data", f"{filename}.pkl")

    # Check if model exists and we want to load it
    if os.path.exists(path) and not force_rerun:
        with open(path, "rb") as f:
            saved = pickle.load(f)
        return saved

    # Apply model_spec to which_vars
    reg_vars.which_vars = {
        getattr(reg_vars, name): include for name, include in spec_dict.items()
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

    # Estimate regression model
    model = RegressionFor(reg_vars).parallel_estimation(df_for, prior_columns)
    model.name = filename

    # Safe save model output
    safe_save_dataframe(model)

    return model


def run_or_load_rbm(
    name: str,
    spec_dict: dict,
    est_vars: ForEstVars,
    agent_vars: AgentVars,
    df_rbm: pd.DataFrame,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """Estimates the RBM or loads a saved model if it exists and meets the specified requirements.

    Parameters
    ----------
    name : str
        The name of the RBM, used for identifying saved files.
    spec_dict : dict
        A dictionary specifying the RBM parameters and configurations.
    est_vars : ForEstVars
        A custom object that holds estimation variables and parameters required for model computation.
    agent_vars : AgentVars
        A custom object that holds agent variables and parameters required for model computation.
    df_rbm : pandas.DataFrame
        Data frame containing the input data used to run the model.
    force_rerun : bool, optional
        If True, forces re-computation of the model even if a saved version exists, by default False.

    Returns
    -------
    pd.DataFrame
        RBM data frame containing the results of the model.
    """

    # Get model hash
    rbm_hash = get_hash(spec_dict)

    # Create file name
    filename = os.path.join(f"{name}_{est_vars.n_sp}sp_{rbm_hash}")
    path = os.path.join("for_data", f"{filename}.pkl")

    # Check if model exists and we want to load it
    if os.path.exists(path) and not force_rerun:
        with open(path, "rb") as f:
            saved = pickle.load(f)
        return saved

    # Apply model_spec to which_vars
    est_vars.which_vars = {
        getattr(est_vars, name): include for name, include in spec_dict.items()
    }

    # Call AlEstimation object
    al_estimation = ForEstimation(est_vars)

    # Estimate parameters and save data
    model = al_estimation.parallel_estimation(df_rbm, agent_vars)
    model.name = filename

    safe_save_dataframe(model)

    return model


def fisher_z_median(rs: list) -> float:
    """Calculate the median of Fisher z-transformed correlation coefficients.

    Parameters
    ----------
    rs : list
        Correlation coefficients.

    Returns
    -------
    float
        Median of the Fisher z-transformed values.

    Raises
    ------
    ValueError
        If any correlation values are non-finite (NaN or infinite).
    """

    # Check all values are finite
    rs_array = np.asarray(rs)
    if not np.all(np.isfinite(rs_array)):
        n_invalid = np.sum(~np.isfinite(rs_array))
        raise ValueError(
            f"Found {n_invalid} non-finite correlation value(s). "
            "All correlations must be finite (not NaN or infinite)."
        )

    # Transform to Fisher z-scores
    zs = [np.arctanh(np.clip(r, -0.999999, 0.999999)) for r in rs_array]

    return float(np.median(zs))


def perm_pval(z_value_obs: float, z_values_perm: np.ndarray, tail: str = "+") -> float:
    """Compute permutation-based p-value.

    Parameters
    ----------
    z_value_obs : float
        Z-value of observed correlation.
    z_values_perm
        Z-values of permuted correlations.
    tail : str
        Type of test, either "+" (one-sided positive), "-" (one-sided negative), or "two-sided" (two-sided).

    Returns
    -------
    float
        Permutation-based p-value.
    """

    # Check if observed z-value is finite
    if not np.isfinite(z_value_obs):
        raise ValueError("Observed z-value must be finite.")

    # Check if permuted z-values are finite
    z_values_perm = np.asarray(z_values_perm, dtype=float)
    if not np.all(np.isfinite(z_values_perm)):
        raise ValueError("Permuted z-values must be finite.")

    # Number of permutations
    n_perm = z_values_perm.size

    # Run specified test
    if tail == "+":
        extremes = np.sum(z_values_perm >= z_value_obs)
    elif tail == "-":
        extremes = np.sum(z_values_perm <= z_value_obs)
    elif tail == "two-sided":
        extremes = np.sum(abs(z_values_perm) >= abs(z_value_obs))
    else:
        raise ValueError("Tail must be '+', '-', or 'two-sided'.")
    return (1 + extremes) / (n_perm + 1)


def sca_wrapper(
    sca_folder: str,
    df_questionnaires: pd.DataFrame,
    df_for: pd.DataFrame,
    df_rbm: pd.DataFrame,
    which_var_quest: list,
    show_validation=True,
    force_rerun=False,
    which_analysis_str: str = "sca",
) -> None:
    """Wrapper function for the specification curve analysis.

    Parameters
    ----------
    sca_folder : str
        Folder in which the specification results will be stored.
    df_questionnaires : pd.DataFrame
        Data frame containing questionnaire scores.
    df_for : pd.DataFrame
        Data frame containing input required for regression analysis.
    df_rbm : pd.DataFrame
        Data frame containing input required for RBM analysis.
    which_var_quest : list
        Questionnaire variable(s) of interest
    show_validation : bool, optional
        Shows model validation plots, by default True
    force_rerun : bool, optional
        Forces re-computation of the model even if a saved version exists, by default False.
    which_analysis_str : str, optional
        String for naming the analysis, by default "sca"

    Returns
    -------
    None
        This function does not return any value.
    """

    n_subj_quest = len(np.unique(df_questionnaires["subj_num"]))
    n_subj_reg = len(np.unique(df_for["subj_num"]))
    n_subj_rbm = len(np.unique(df_rbm["subj_num"]))

    # -----------------
    # Create SCA object
    # -----------------

    sca = SpecificationCurveAnalysis()
    sca.expected_n_subj = n_subj_quest
    sca.which_analysis_str = which_analysis_str

    # -------------------------
    # Regression specifications
    # -------------------------

    # Determine if we need to rerun the models even if they already exist
    reg_vars = RegVars()
    reg_vars.n_subj = n_subj_reg  # number of subjects
    reg_vars.n_ker = 4  # number of kernels for estimation
    reg_vars.n_sp = 50  # number of random starting points
    reg_vars.rand_sp = True  # use random starting points
    reg_vars.use_prior = False

    # Get regression specifications
    regression_specs = get_regression_specs(sca_folder)

    # Initialize variable
    all_bic = np.full([n_subj_reg, len(regression_specs.items())], np.nan)
    all_lr = np.full([n_subj_reg, len(regression_specs.items())], np.nan)
    all_names = list()

    # Run or load models
    for i, (name, spec) in enumerate(regression_specs.items()):

        # Extract the actual regression specification
        current_spec = spec[name].copy()

        result = run_or_load_regression(
            name, current_spec, reg_vars, df_for, force_rerun=force_rerun
        )
        # Store results for model comparison
        all_bic[:, i] = result["BIC"]
        all_lr[:, i] = result["beta_1"]
        all_names.append(name)

    # Fast model comparison
    if show_validation:
        sca.fast_model_comp(all_lr, all_bic)

    # ---------------
    # Model selection
    # ---------------

    def model_selection(bic_array: np.ndarray, name_list: list) -> list:
        """ This performs a model BIC-based model selection for the SCA.

        Parameters
        ----------
        bic_array : np.ndarray
            Array of BIC values.
        name_list : list
            List of model names.

        Returns
        -------
        list
            Selected models
        """

        # Todo: note that this is a preliminary function that has not been
        #   validated or tested yet.

        # Find the maximum BIC value for each subject across the models
        subject_max_bic = np.max(bic_array, axis=1)

        # Calculate delta BIC for each subject
        delta_bic = (
            subject_max_bic[:, np.newaxis] - bic_array
        )

        # Identify selection threshold
        is_acceptable = delta_bic <= 10.0

        # Calculate average number of included models
        model_inclusion_rates = np.mean(is_acceptable, axis=0)

        # Get the indices of the models that survive the threshold (e.g., >= 20%)
        inclusion_threshold = 0.20
        surviving_model_indices = np.where(
            model_inclusion_rates >= inclusion_threshold
        )[0]

        # Put everything into a list with names
        included_model_list = [name_list[i] for i in surviving_model_indices]

        return included_model_list

    included_model_names = model_selection(all_bic, all_names)

    # Rebuild the dictionary, keeping only keys that end with an included model name
    filtered_regression_specs = {
        reg_name: reg_spec
        for reg_name, reg_spec in regression_specs.items()
        # Check if any of the included model names are at the end of the full path key
        if any(reg_name.endswith(model_name) for model_name in included_model_names)
    }

    # ---------------------
    # Estimate RBM directly
    # ---------------------

    # Call AgentVars Object
    agent_vars = AgentVars()
    agent_vars.max_x = 2 * np.pi

    # Call AlEstVars object
    est_vars = ForEstVars()
    est_vars.n_subj = n_subj_rbm  # number of subjects
    est_vars.n_ker = 4  # number of kernels for estimation
    est_vars.n_sp = 10  # number of random starting points
    est_vars.rand_sp = True  # use random starting points
    est_vars.use_prior = (
        True  # use weakly informative prior for uncertainty underestimation
    )

    # Get RBM specifications
    rbm_specs = get_rbm_specs(sca_folder)

    # Initialize variables
    all_bic = np.full([n_subj_reg, len(rbm_specs.items())], np.nan)
    all_h = np.full([n_subj_reg, len(rbm_specs.items())], np.nan)
    all_names = list()

    # Run or load models
    for i, (name, spec) in enumerate(rbm_specs.items()):

        result = run_or_load_rbm(
            name, spec, est_vars, agent_vars, df_rbm, force_rerun=force_rerun
        )

        # Store results for model comparison
        all_bic[:, i] = result["BIC"]
        all_h[:, i] = result["h"]
        all_names.append(name)

    # Fast model comparison
    if show_validation:
        sca.fast_model_comp(all_h, all_bic, voi_name="Hazard Rate")

    included_model_names = model_selection(all_bic, all_names)

    # Rebuild the dictionary, keeping only keys that end with an included model name
    filtered_rbm_specs = {
        rbm_name: rbm_spec
        for rbm_name, rbm_spec in rbm_specs.items()
        # Check if any of the included model names are at the end of the full path key
        if any(rbm_name.endswith(model_name) for model_name in included_model_names)
    }
    # --------------------------------
    # Run specification curve analysis
    # --------------------------------

    # Ensure that we have an empty folder w/o any previous results
    os.makedirs("for_data/sca_temp", exist_ok=True)
    for f in os.listdir("for_data/sca_temp/"):
        os.remove(os.path.join("for_data/sca_temp/", f))

    # Evaluate all specifications
    sca.run_sca(
        filtered_regression_specs,
        reg_vars,
        filtered_rbm_specs,
        est_vars,
        agent_vars,
        df_for,
        df_rbm,
        df_questionnaires,
        which_var_quest=which_var_quest,
    )

    # ----------------
    # Permutation test
    # ----------------

    p_t1 = sca.run_permutation_test()

    # --------
    # Plotting
    # --------

    sca.plot_sca(p_t1)


def get_regression_specs(sca_folder: str) -> dict[str, dict[str, bool]]:
    """Returns the regression specifications for the SCA.

    Parameters
    ----------
    sca_folder : str
        Folder in which the specification results will be stored.

    Returns
    -------
    dict[str, dict[str, bool]]
        Dictionary containing the regression specifications for the SCA.

    """

    regression_specs = {
        sca_folder
        + "regression_11": {
            sca_folder
            + "regression_11": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": True,  # omega_t
                "beta_3": True,  # tau_t
                "beta_4": False,  # alpha_t
                "beta_5": False,  # r_t
                "beta_6": False,  # sigma_t
                "beta_7": False,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": False},
        },
        sca_folder
        + "regression_12": {
            sca_folder
            + "regression_12": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": True,  # omega_t
                "beta_3": True,  # tau_t
                "beta_4": False,  # alpha_t
                "beta_5": True,  # r_t
                "beta_6": False,  # sigma_t
                "beta_7": False,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": False},
        },
        sca_folder
        + "regression_13": {
            sca_folder
            + "regression_13": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": True,  # omega_t
                "beta_3": True,  # tau_t
                "beta_4": False,  # alpha_t
                "beta_5": False,  # r_t
                "beta_6": True,  # sigma_t
                "beta_7": False,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": False},
        },
        sca_folder
        + "regression_14": {
            sca_folder
            + "regression_14": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": True,  # omega_t
                "beta_3": True,  # tau_t
                "beta_4": False,  # alpha_t
                "beta_5": False,  # r_t
                "beta_6": False,  # sigma_t
                "beta_7": True,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": False},
        },
        sca_folder
        + "regression_15": {
            sca_folder
            + "regression_15": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": True,  # omega_t
                "beta_3": True,  # tau_t
                "beta_4": False,  # alpha_t
                "beta_5": True,  # r_t
                "beta_6": True,  # sigma_t
                "beta_7": False,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": False},
        },
        sca_folder
        + "regression_16": {
            sca_folder
            + "regression_16": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": True,  # omega_t
                "beta_3": True,  # tau_t
                "beta_4": False,  # alpha_t
                "beta_5": True,  # r_t
                "beta_6": False,  # sigma_t
                "beta_7": True,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": False},
        },
        sca_folder
        + "regression_17": {
            sca_folder
            + "regression_17": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": True,  # omega_t
                "beta_3": True,  # tau_t
                "beta_4": False,  # alpha_t
                "beta_5": False,  # r_t
                "beta_6": True,  # sigma_t
                "beta_7": True,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": False},
        },
        sca_folder
        + "regression_18": {
            sca_folder
            + "regression_18": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": True,  # omega_t
                "beta_3": True,  # tau_t
                "beta_4": False,  # alpha_t
                "beta_5": True,  # r_t
                "beta_6": True,  # sigma_t
                "beta_7": True,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": False},
        },
        sca_folder
        + "regression_21": {
            sca_folder
            + "regression_21": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": False,  # omega_t
                "beta_3": False,  # tau_t
                "beta_4": True,  # alpha_t
                "beta_5": False,  # r_t
                "beta_6": False,  # sigma_t
                "beta_7": False,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": True},
        },
        sca_folder
        + "regression_22": {
            sca_folder
            + "regression_22": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": False,  # omega_t
                "beta_3": False,  # tau_t
                "beta_4": True,  # alpha_t
                "beta_5": True,  # r_t
                "beta_6": False,  # sigma_t
                "beta_7": False,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": True},
        },
        sca_folder
        + "regression_23": {
            sca_folder
            + "regression_23": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": False,  # omega_t
                "beta_3": False,  # tau_t
                "beta_4": True,  # alpha_t
                "beta_5": False,  # r_t
                "beta_6": True,  # sigma_t
                "beta_7": False,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": True},
        },
        sca_folder
        + "regression_24": {
            sca_folder
            + "regression_24": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": False,  # omega_t
                "beta_3": False,  # tau_t
                "beta_4": True,  # alpha_t
                "beta_5": False,  # r_t
                "beta_6": False,  # sigma_t
                "beta_7": True,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": True},
        },
        sca_folder
        + "regression_25": {
            sca_folder
            + "regression_25": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": False,  # omega_t
                "beta_3": False,  # tau_t
                "beta_4": True,  # alpha_t
                "beta_5": True,  # r_t
                "beta_6": True,  # sigma_t
                "beta_7": False,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": True},
        },
        sca_folder
        + "regression_26": {
            sca_folder
            + "regression_26": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": False,  # omega_t
                "beta_3": False,  # tau_t
                "beta_4": True,  # alpha_t
                "beta_5": True,  # r_t
                "beta_6": False,  # sigma_t
                "beta_7": True,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": True},
        },
        sca_folder
        + "regression_27": {
            sca_folder
            + "regression_27": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": False,  # omega_t
                "beta_3": False,  # tau_t
                "beta_4": True,  # alpha_t
                "beta_5": False,  # r_t
                "beta_6": True,  # sigma_t
                "beta_7": True,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": True},
        },
        sca_folder
        + "regression_28": {
            sca_folder
            + "regression_28": {
                "beta_0": True,  # intercept
                "beta_1": True,  # delta_t
                "beta_2": False,  # omega_t
                "beta_3": False,  # tau_t
                "beta_4": True,  # alpha_t
                "beta_5": True,  # r_t
                "beta_6": True,  # sigma_t
                "beta_7": True,  # catch-trial * PE
                "beta_8": False,  # catch-trial * EE
                "omikron_0": True,  # motor noise
                "omikron_1": True,  # learning-rate noise
                "lambda_0": False,  # pers intercept when comb w/ lambda_1 or overall probability
                "lambda_1": False,  # perseveration slope (when combined with lambda_1)
            },
            "dependent_variable": {"beta_1": True, "beta_4": True},
        },
    }

    return regression_specs


def get_rbm_specs(sca_folder) -> dict[str, dict[str, bool]]:
    """Returns the RBM specifications for the SCA.

    Parameters
    ----------
    sca_folder : str
        Folder in which the specification results will be stored.

    Returns
    -------
    dict[str, dict[str, bool]]
        Dictionary containing the RBM specifications.
        Dictionary containing the RBM specifications.
    """
    rbm_specs = {
        sca_folder
        + "rbm_11": {
            "omikron_0": True,  # motor noise
            "omikron_1": True,  # learning-rate noise
            "lambda_0": False,  # perseveration intercept
            "lambda_1": False,  # perseveration slope
            "h": True,  # hazard rate
            "s": False,  # surprise sensitivity
            "u": False,  # uncertainty underestimation
            "sigma_H": False,  # catch trials
        },
        sca_folder
        + "rbm_12": {
            "omikron_0": True,  # motor noise
            "omikron_1": True,  # learning-rate noise
            "lambda_0": False,  # perseveration intercept
            "lambda_1": False,  # perseveration slope
            "h": True,  # hazard rate
            "s": True,  # surprise sensitivity
            "u": False,  # uncertainty underestimation
            "sigma_H": False,  # catch trials
        },
        sca_folder
        + "rbm_13": {
            "omikron_0": True,  # motor noise
            "omikron_1": True,  # learning-rate noise
            "lambda_0": False,  # perseveration intercept
            "lambda_1": False,  # perseveration slope
            "h": True,  # hazard rate
            "s": False,  # surprise sensitivity
            "u": True,  # uncertainty underestimation
            "sigma_H": False,  # catch trials
        },
        sca_folder
        + "rbm_14": {
            "omikron_0": True,  # motor noise
            "omikron_1": True,  # learning-rate noise
            "lambda_0": False,  # perseveration intercept
            "lambda_1": False,  # perseveration slope
            "h": True,  # hazard rate
            "s": False,  # surprise sensitivity
            "u": False,  # uncertainty underestimation
            "sigma_H": True,  # catch trials
        },
        sca_folder
        + "rbm_15": {
            "omikron_0": True,  # motor noise
            "omikron_1": True,  # learning-rate noise
            "lambda_0": False,  # perseveration intercept
            "lambda_1": False,  # perseveration slope
            "h": True,  # hazard rate
            "s": True,  # surprise sensitivity
            "u": True,  # uncertainty underestimation
            "sigma_H": False,  # catch trials
        },
        sca_folder
        + "rbm_16": {
            "omikron_0": True,  # motor noise
            "omikron_1": True,  # learning-rate noise
            "lambda_0": False,  # perseveration intercept
            "lambda_1": False,  # perseveration slope
            "h": True,  # hazard rate
            "s": True,  # surprise sensitivity
            "u": False,  # uncertainty underestimation
            "sigma_H": True,  # catch trials
        },
        sca_folder
        + "rbm_17": {
            "omikron_0": True,  # motor noise
            "omikron_1": True,  # learning-rate noise
            "lambda_0": False,  # perseveration intercept
            "lambda_1": False,  # perseveration slope
            "h": True,  # hazard rate
            "s": False,  # surprise sensitivity
            "u": True,  # uncertainty underestimation
            "sigma_H": True,  # catch trials
        },
        sca_folder
        + "rbm_18": {
            "omikron_0": True,  # motor noise
            "omikron_1": True,  # learning-rate noise
            "lambda_0": False,  # perseveration intercept
            "lambda_1": False,  # perseveration slope
            "h": True,  # hazard rate
            "s": True,  # surprise sensitivity
            "u": True,  # uncertainty underestimation
            "sigma_H": True,  # catch trials
        },
    }

    return rbm_specs


# ----------------------------------------
# Global factor-analysis-related functions
# ----------------------------------------


def fa_candidates(param_space: dict) -> Generator:
    """This function iterates through the factor-analysis specification for evaluation.

    The function only yields combinations that satisfy the condition "check_fa_spec(fa)".

    Parameters
    ----------
    param_space : dict
         A mapping from parameter names to their possible values.

    Yields
    ------
    dict
        Parameter combination that passes the check.
    """

    # Extract keys (parameter names) of parameter space
    keys = list(param_space)

    # Cycle through values of our parameter space
    for vals in itertools.product(*[param_space[k] for k in keys]):

        # Current specification
        fa = dict(zip(keys, vals))

        # Check if current specification is valid and if so, generate specification
        if check_fa_spec(fa):
            yield fa


def check_fa_spec(fa: dict) -> bool:
    """This function evaluates whether a factor-analysis combination is valid or not.

    Parameters
    ----------
    fa : dict
        Currently proposed combination of factor-analysis properties.

    Returns
    -------
    bool
        Indicates if the factor-analysis combination is valid or not.
    """

    # We can't do bifactor with varimax rotation: flag as invalid
    if fa.get("analysis_type") == "bifactor" and fa.get("rotation") == "varimax":
        return False

    return True
