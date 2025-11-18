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
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from allinpy import cm2inch
from matplotlib.colors import ListedColormap
from numpy.random import default_rng
from tqdm import tqdm

from FOR_1_Paper.for_utilities import safe_save_dataframe
from FOR_1_Paper.regression.RegressionFor import RegressionFor


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


def check_fa_pool(pool: list[dict[str, str]], all_expected=False) -> None:
    """Checks the validity and presence of required FA files in a specified folder.

    This function performs two checks:
    1. Verifies that all expected FA files based on the 'pool' are present in the
       specified folder.
    2. Optionally verifies that all files in the folder match expected hash codes
       when `all_expected` is set to True.

    Parameters
    ----------
    pool : list[dict[str, str]]
        List of dictionaries containing FA data used to generate hash codes.
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

    # Loop over all FAs and compute hash codes
    for fa in pool:

        # Get hash code
        fa_hash = get_hash(fa)
        hash_list.append(fa_hash)

        # Check if file exists in data folder
        folder = Path("for_data")
        found = any(folder.glob("sca_fa_" + fa_hash + ".pkl"))

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


def build_specs_with_vars(
    pool: list, var_names: list, var_rules: dict, spec_prefix="spec"
) -> dict:
    """This function builds factor-analysis combinations based on the constraints.

    Parameters
    ----------
    pool : list
        Set of factor analyses.
    var_names : list
        Names of rules to be applied.
    var_rules : dict
        Rules to be applied.
    spec_prefix : str
        Prefix for the specification names.

    Returns
    -------
    dict
        Dictionary of specifications.
    """

    specs = {}
    per_flag = {f: 0 for f in var_names}

    # Cycle over factor analyses
    for fa in pool:

        # Extract hash code
        fa_hash = get_hash(fa)

        # Cycle over variables
        for var in var_names:

            # Skip if variable does not pass the rule
            if not passes_variable_rules(fa, var, var_rules):
                continue

            # Add to set of specifications
            spec = {f: (f == var) for f in var_names}
            spec["fa"] = deepcopy(fa)
            specs[f"{spec_prefix}_{var}_{fa_hash}"] = spec
            per_flag[var] += 1

    return specs


def passes_variable_rules(fa_spec: dict, variable_name: str, var_rules: dict) -> bool:
    """Determines whether the variable rules are satisfied for a given factor-analysis
    specification and variable name.

    Parameters
    ----------
    fa_spec : dict
        A dictionary representing the factor-analysis specification.
    variable_name : str
        The name of the variable for which rules need to be validated.
    var_rules : dict
        A dictionary mapping variable names to lists of callable rule functions. Each
        callable takes `fa_spec` as input and returns a boolean indicating whether
        the rule is satisfied.

    Returns
    -------
    bool
        True if all rules for the variable are satisfied; False otherwise.
    """
    # Get rules for this variable (empty list if variable not found)
    rules_for_variable = var_rules.get(variable_name, [])

    # Check each rule
    for rule_function in rules_for_variable:

        # Check for current factor-analysis specification if the specified rule applies
        if not rule_function(fa_spec):
            return False

    # Return True if all rules have been satisfied
    return True


def run_sca(
    regression_specs: dict,
    analysis_specs: dict,
    reg_vars: "RegVars",
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
        DataFrame containing input required for regression analysis.
    which_factor : pd.DataFrame
        DataFrame indicating which factor is used for the specifications.
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
        - all_betas : list of pd.DataFrame
          List of regression coefficients (e.g., learning rates), indexed by subject.
        - all_results : list of dict
          List containing analysis results (e.g., correlation coefficients and p-values)
          for each specification combination.
    """

    # Initialize output lists
    all_factors = []
    all_betas = []
    all_results = []

    # Initialize counter for significant results
    significance_counter = 0

    # Initialize counter for specifications
    counter = 0

    # Cycle over all specifications and compute correlation
    # -----------------------------------------------------

    # Regression specifications
    for reg_name, reg_spec in regression_specs.items():

        # Compute or load regression result (if pre-computed)
        df_reg = run_or_load_model(
            reg_name, reg_spec, reg_vars, df_for, force_rerun=force_rerun
        )

        # Factor analysis specifications
        for analysis_name, analysis_spec in analysis_specs.items():

            # Update counter
            counter += 1

            # Select matching subjects
            df_sca_fa, df_reg, fa_hash = filter_subjects(analysis_spec, df_reg)

            # Compute correlation between factor scores and regression coefficients
            analysis_result, df_sca_fa, df_reg, factor_name = correlate_reg_fa(
                which_factor, fa_hash, df_sca_fa, df_reg, which_var
            )

            # Count number of single-test significant results
            if analysis_result.get("p_value", 1.0) <= 0.05:
                significance_counter += 1

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
            spec_result.name = f"for_data/sca/sca_{sca_hash}.pkl"  # result_path
            spec_result.name = f"sca_{sca_hash}.pkl"
            safe_save_dataframe(
                spec_result, data_dir="for_data/sca/", print_action=False
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
            all_betas.append(betas_vector)
            all_results.append(analysis_result)

    print(f"Number of significant results: {significance_counter}")

    return all_factors, all_betas, all_results


def correlate_reg_fa(
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
    factor_name = which_factor[which_factor["fa_hash"] == fa_hash]["max_index"].iloc[0]

    # Select variable of interest for correlation analysis
    df_fa_voi = df_sca_fa[factor_name].to_numpy()
    df_reg_voi = df_reg[which_var].to_numpy()

    # Sanity check if we have enough overlapping subjects
    assert (
        len(df_fa_voi) == len(df_reg_voi) and len(df_fa_voi) == 65
    ), "Not enough overlapping subjects"

    # Compute correlation
    r, p = stats.pearsonr(df_fa_voi, df_reg_voi)

    # Todo: this needs to fit all analyses ultimately
    analysis_result = {
        "effect": r,
        "p_value": p,
    }

    return analysis_result, df_sca_fa, df_reg, factor_name


def filter_subjects(
    analysis_spec: dict, df_data: pd.DataFrame
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
    curr_file_name = f"for_data/sca_fa_{fa_hash}.pkl"
    df_sca_fa = pd.read_pickle(curr_file_name)

    # Unify subject column name
    if "ID" in df_sca_fa.columns:
        pass
    elif "subj_num" in df_sca_fa.columns:
        df_sca_fa = df_sca_fa.rename(columns={"subj_num": "ID"})
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


def run_or_load_model(
    name: str,
    spec_dict: dict,
    reg_vars: "RegVars",
    df_for: pd.DataFrame,
    force_rerun: bool = False,
):
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
    RegressionFor
        Regression model object.
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


def run_sca_permutation_test(
    all_factors: list,
    all_betas: list,
    all_results: list,
    n_perm: int = 1000,
    expected_n_subj: int = 65,
) -> float:
    """Run permutation test for specification curve analysis.

    Parameters
    ----------
    all_factors : list
        All factor scores from the participants.
    all_betas : list
        All betas from the participants.
    all_results : list
        Correlation results, primarily for validation.
    n_perm : int, optional
        Number of permutations, by default 1000.
    expected_n_subj : int, optional
        Expected number of subjects, by default 65.

    Returns
    -------
    float
        Permutation-test-based p-value.
    """

    # Extract IDs from factor scores
    common_ids = set(all_factors[0].index)
    common_ids = sorted(common_ids)

    # Sanity check if we have enough overlapping subjects
    assert len(common_ids) == expected_n_subj, "Not enough overlapping subjects"

    # Put factors and betas into matrices for permutation testing
    all_factors_matrix = np.column_stack(
        [x.reindex(common_ids).to_numpy() for x in all_factors]
    )
    all_betas_matrix = np.column_stack(
        [y.reindex(common_ids).to_numpy() for y in all_betas]
    )

    # Compute correlation between factors and betas (not permuted)
    rs_obs = []
    for j in range(all_factors_matrix.shape[1]):
        r, _ = stats.pearsonr(all_factors_matrix[:, j], all_betas_matrix[:, j])
        rs_obs.append(r)

        # Check whether r-values are consistent with original results
        r_prev = all_results[j]["effect"]
        assert r == r_prev, "Correlation mismatch"

    # Compute Fisher z-score as test statistic
    z_value_obs = fisher_z_median(rs_obs)

    # Permutation test: Same subject permutation applied to all columns of all_betas_matrix
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
        perm = rng.permutation(all_factors_matrix.shape[0])

        # Apply to every spec simultaneously
        all_factors_perm = all_factors_matrix[perm, :]

        # Initialize list with correlations
        r_list = []

        # Cycle over factor analysis permutations
        # todo: add freedman-lane when regression is fully implemented
        for j in range(all_factors_perm.shape[1]):

            # Compute correlation between permuted factors and betas
            r, _ = stats.pearsonr(all_factors_perm[:, j], all_betas_matrix[:, j])
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
    plt.savefig("figures/p_value_hist.png", dpi=300)

    return p_value


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


def plot_sca(
    var_rule_ids: list[str],
    p_T1: float,
    ylabel: str = "Effect Size Fixed Learning Rate",
):
    """Plot the specification curve analysis results.

    Parameters
    ----------
    var_rule_ids : list[str]
        Name of the current sca analysis rule.
    p_T1 : float
        Permutation-based p-value.
    ylabel : str, optional
        Y-axis label, by default "Effect Size Fixed Learning Rate"

    Returns
    -------
    None
        This function does not return any value.
    """

    # ------------
    # Get SCA data
    # ------------

    # Get all SCA files
    result_files = glob.glob("for_data/sca/*.pkl")

    # Initialize list summarizing all specifications
    summary_rows = []

    # Cycle over all SCAs to extract exact specifications
    for file in result_files:

        # Load current file and add to summary list
        result = pd.read_pickle(file)  # ← `file` is a path string
        row = result.iloc[0]  # single-row DataFrame
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
        data={"norm_sep": norm_sep, "norm_comb": norm_comb, "covariates": covariates}
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
    step = max(1, len(plot_df) // 30)

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
    #     linewidths=0,  # no gridlines → crisp black/white
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
    ax_tile.set_xticklabels(
        x_shifted[::step].astype(int), rotation=45
    )

    sns.despine()
    plt.tight_layout()

    # Save figure
    output_file = "figures/multiverse_" + var_rule_ids[0] + ".png"
    plt.savefig(output_file, dpi=400)


def sca_fast_model_comp(bic: list, learning_rate: list, all_lr: np.ndarray):
    """Runs a quick-and-dirty model comparison for SCA.

    Parameters
    ----------
    bic : list
        List with BIC values.
    learning_rate : list
        List with learning rates.
    all_lr : np.ndarray
        All learning rates for single-subject plots.

    Returns
    -------
    None
        This function does not return any value.
    """

    # Plot BIC for quick and dirty model comparison
    plt.figure()
    plt.bar(np.arange(len(bic)), bic)

    # Plot learning rates to see if models are similar
    plt.figure()
    plt.bar(np.arange(len(learning_rate)), learning_rate)
    ax = plt.gca()

    # Convert to long format DataFrame for stripplot
    lr_df = pd.DataFrame(all_lr, columns=[f"Model_{i}" for i in range(len(bic))])
    lr_long = lr_df.melt(var_name="model", value_name="LR")
    lr_long["model_idx"] = lr_long["model"].str.extract("(\d+)").astype(int)

    # Add single points
    sns.stripplot(
        data=lr_long, x="model_idx", y="LR", color="k", alpha=0.7, size=2, ax=ax
    )
