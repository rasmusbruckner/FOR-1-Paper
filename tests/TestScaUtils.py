"""Unit tests for sca_utils.py"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from FOR_1_Paper.regression.ForRegVars import RegVars
from FOR_1_Paper.sca import sca_utils
from FOR_1_Paper.sca.sca_utils import (build_specs_with_vars, check_fa_pool,
                                       check_fa_spec, correlate_reg_fa,
                                       fa_candidates, filter_subjects,
                                       fisher_z_median, get_hash,
                                       passes_variable_rules, perm_pval,
                                       run_or_load_model, run_sca,
                                       run_sca_permutation_test)


def test_fa_candidates():
    """Tests the factor-analysis candidate generation, where bifactor and varimax are not allowed."""

    # Simple parameter space
    param_space = {
        "analysis_type": ["simple", "bifactor"],
        "rotation": ["oblimin", "varimax"],
    }

    actual_pool = list(fa_candidates(param_space))

    expected_pool = [
        {"analysis_type": "simple", "rotation": "oblimin"},
        {"analysis_type": "simple", "rotation": "varimax"},
        {"analysis_type": "bifactor", "rotation": "oblimin"},
    ]

    assert actual_pool == expected_pool


def test_fa_spec():
    """Tests the factor-analysis specification checker, which returns "False" for bifactor with varimax rotation."""

    fa = {"analysis_type": "simple", "rotation": "oblimin"}
    allowed = check_fa_spec(fa)
    assert allowed

    fa = {"analysis_type": "bifactor", "rotation": "varimax"}
    allowed = check_fa_spec(fa)
    assert not allowed


def _fname(h: str) -> str:
    """Helper function that creates a file name from a hash."""

    return f"sca_fa_{h}.pkl"


def _touch(folder: Path, name: str) -> None:
    """Helper function that creates an empty file in the given folder with the given name."""

    (folder / name).write_text("{}", encoding="utf-8")


def test_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Tests the case that everything is complete."""

    # Monkeypatch Path to return tmp_path when instantiated with "for_data"
    def mock_path(path_str):
        """Mock function for Path."""

        if path_str == "for_data":
            return tmp_path
        return Path(path_str)

    monkeypatch.setattr("FOR_1_Paper.sca.sca_utils.Path", mock_path)

    fa1 = {"factor_method": "ml", "rotation": "varimax", "n_factors": "3"}
    fa2 = {"factor_method": "minres", "rotation": "oblimin", "n_factors": "4"}

    # Cycle over fas to create temporary files for testing
    for fa in (fa1, fa2):

        # Create temporary file for testing
        _touch(tmp_path, _fname(get_hash(fa)))

    # Run to see if it does not raise error
    check_fa_pool([fa1, fa2], all_expected=False)


def test_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Tests the case that file is not found."""

    # Monkeypatch Path to return tmp_path when instantiated with "for_data"
    def mock_path(path_str):
        """Mock function for Path."""

        if path_str == "for_data":
            return tmp_path
        return Path(path_str)

    monkeypatch.setattr("FOR_1_Paper.sca.sca_utils.Path", mock_path)

    fa = {"factor_method": "ml", "rotation": "varimax", "n_factors": "3"}

    # We haven't created a file, so this should raise an error
    with pytest.raises(FileNotFoundError):
        check_fa_pool([fa], all_expected=False)


def test_all_expected_flags_extras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Tests the case that all expected files are present, but there are extras."""

    # Monkeypatch Path to return tmp_path when instantiated with "for_data"
    def mock_path(path_str):
        """Mock function for Path."""

        if path_str == "for_data":
            return tmp_path
        return Path(path_str)

    monkeypatch.setattr("FOR_1_Paper.sca.sca_utils.Path", mock_path)

    fa = {"factor_method": "ml", "rotation": "varimax", "n_factors": "3"}

    # Create a file that the function expects
    _touch(tmp_path, _fname(get_hash(fa)))

    # Create a file that is not expected
    _touch(tmp_path, _fname("0" * 32))

    # Should raise an error because there are extras
    with pytest.raises(FileNotFoundError):
        check_fa_pool([fa], all_expected=True)


def test_get_hash():
    """Tests the function that computes the hash of a factor-analysis specification."""

    fa = {"factor_method": "ml", "rotation": "varimax", "n_factors": "3"}
    h = get_hash(fa)

    assert h == "a0e998ac99d5c92d6f598489303eb377"


def test_build_specs_with_vars_all_included():
    """Tests the case where all variable rules apply."""

    # Parameter space of factor analysis
    param_space = {
        "analysis_type": ["simple", "bifactor"],
        "data_type": ["big_data"],
    }

    # Analysis specifications based on our parameter space
    pool = list(fa_candidates(param_space))

    var_rule_ids = ["psychosis"]

    var_rules = {
        "psychosis": [
            lambda fa: True,
        ],
    }

    analysis_specs = build_specs_with_vars(pool, var_rule_ids, var_rules)

    expected_specs = {
        "spec_psychosis_7d367668b66d4db14869a79a535b8f63": {
            "fa": {"analysis_type": "bifactor", "data_type": "big_data"},
            "psychosis": True,
        },
        "spec_psychosis_e196ec8be4c8c80319b59fc1c0e92b4b": {
            "fa": {"analysis_type": "simple", "data_type": "big_data"},
            "psychosis": True,
        },
    }
    assert analysis_specs == expected_specs


def test_build_specs_with_vars_some_included():
    """Tests the case where only some variable rules apply."""

    # Parameter space of factor analysis
    param_space = {
        "analysis_type": ["simple", "bifactor"],
        "data_type": ["big_data"],
    }

    # Analysis specifications based on our parameter space
    pool = list(fa_candidates(param_space))

    var_rule_ids = ["g_only"]

    # Only applies to bifactor
    var_rules = {
        "g_only": [lambda fa: fa["analysis_type"] == "bifactor"],
    }

    analysis_specs = build_specs_with_vars(pool, var_rule_ids, var_rules)

    expected_specs = {
        "spec_g_only_7d367668b66d4db14869a79a535b8f63": {
            "fa": {"analysis_type": "bifactor", "data_type": "big_data"},
            "g_only": True,
        }
    }

    assert analysis_specs == expected_specs


def test_passes_variable_rules_all_passed():
    """Tests the case where all variable rules apply."""

    fa = {"analysis_type": "simple", "rotation": "oblimin"}
    var = "psychosis"

    # Always applies
    var_rules = {
        "psychosis": [
            lambda fa: True,
        ],
    }

    passed = passes_variable_rules(fa, var, var_rules)
    assert passed


def test_passes_variable_rules_specific_passed():
    """Tests the case where a specific variable rule applies."""

    fa = {"analysis_type": "bifactor", "rotation": "oblimin"}
    var = "g_only"

    # Only applies to bifactor
    var_rules = {
        "g_only": [lambda fa: fa["analysis_type"] == "bifactor"],
    }

    passed = passes_variable_rules(fa, var, var_rules)
    assert passed


def test_passes_variable_rules_specific_failed():
    """Tests the case where a specific variable rule does not apply."""

    fa = {"analysis_type": "simple", "rotation": "oblimin"}
    var = "g_only"

    # Only applies to bifactor
    var_rules = {
        "g_only": [lambda fa: fa["analysis_type"] == "bifactor"],
    }

    passed = passes_variable_rules(fa, var, var_rules)
    assert not passed


def make_reg_df(n=65, which_var="beta_1"):
    """Creates a mock reg_df dataframe for unit testing."""

    # IDs
    ids = np.arange(1, n + 1)

    # Betas: this will have a perfect correlation with factor analysis:
    # factor = 2*beta + 1
    beta = np.linspace(0, 1, n)
    df = pd.DataFrame({"ID": ids, which_var: beta})
    return df


def make_df_sca_fa_with_id(n=65, factor_name="F1"):
    """Creates a mock dataframe for the factor analysis with ID for unit testing."""

    # IDs
    ids = np.arange(1, n + 1)

    # Factor score: linearly related to beta
    factor = np.linspace(0, 1, n) * 2 + 1
    df = pd.DataFrame({"ID": ids, factor_name: factor})
    return df


def make_df_sca_fa_with_subj_num(n=65, factor_name="F1"):
    """Creates a mock dataframe for the factor analysis with subj_num for unit testing."""

    # Create dataframe
    df = make_df_sca_fa_with_id(n, factor_name)

    # Rename ID to subj_num
    df = df.rename(columns={"ID": "subj_num"})
    return df


def mock_get_hash(obj):
    """Mock function to replace get_hash."""

    # Depending on input return different hashes
    if isinstance(obj, dict) and "fa" in obj:
        return "FAHASH"
    return "SCAHASH"


# Initialize dict for mock save
saved = {}


def mock_save_dataframe(df, data_dir, print_action):
    """Mock function to replace safe_save_dataframe."""

    saved["name"] = getattr(df, "name", None)
    saved["data_dir"] = data_dir
    saved["columns"] = list(df.columns)
    saved["row"] = df.iloc[0].to_dict()


def test_run_sca(monkeypatch, tmp_path):
    """Tests the function that runs the SCA."""

    # Inputs
    regression_specs = {"regression_11": {"reg_param": 1}}
    analysis_specs = {"fa_11": {"fa": {"fa": 3}, "fa_param": "X"}}
    which_var = "beta_1"

    which_factor = pd.DataFrame({"fa_hash": ["FAHASH"], "max_index": ["F1"]})

    # Mock dataframes
    df_reg = make_reg_df(which_var=which_var)
    df_fa = make_df_sca_fa_with_id(factor_name="F1")

    monkeypatch.setattr("FOR_1_Paper.sca.sca_utils.get_hash", mock_get_hash)
    monkeypatch.setattr(
        "FOR_1_Paper.sca.sca_utils.run_or_load_model", lambda *a, **k: df_reg
    )
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa)
    monkeypatch.setattr(
        "FOR_1_Paper.sca.sca_utils.safe_save_dataframe", mock_save_dataframe
    )

    # Call function
    all_factors, all_betas, all_results = run_sca(
        regression_specs=regression_specs,
        analysis_specs=analysis_specs,
        reg_vars=object(),
        df_for=pd.DataFrame(),
        which_factor=which_factor,
        which_var=which_var,
        force_rerun=False,
    )

    # Test length of output
    assert len(all_factors) == 1
    assert len(all_betas) == 1
    assert len(all_results) == 1

    # Test whether factors and betas have the same ID index
    assert all_factors[0].index.equals(all_betas[0].index)

    # Test whether correlation between factors and betas is 1
    assert all_results[0]["effect"] == pytest.approx(1.0, rel=1e-6)

    # Test whether we save the correct output
    assert saved["name"].endswith("sca_SCAHASH.pkl")
    assert saved["data_dir"] == "for_data/sca/"
    assert saved["columns"] == [
        "model_id",
        "reg_param",
        "fa",
        "fa_param",
        "effect",
        "p_value",
    ]
    assert saved["row"] == {
        "effect": 0.9999999999999999,
        "fa": {"fa": 3},
        "fa_param": "X",
        "model_id": "model_1",
        "p_value": 0.0,
        "reg_param": 1,
    }


def test_run_sca_with_subj_num(monkeypatch, tmp_path):
    """Tests the function that runs the SCA with subj_num input."""

    # Inputs
    regression_specs = {"regression_11": {"reg_param": 1}}
    analysis_specs = {"fa_11": {"fa": {"fa": 3}, "fa_param": "X"}}
    which_var = "beta_1"

    which_factor = pd.DataFrame({"fa_hash": ["FAHASH"], "max_index": ["F1"]})

    # Mock dataframes
    df_reg = make_reg_df(which_var=which_var)
    df_fa = make_df_sca_fa_with_subj_num(factor_name="F1")

    monkeypatch.setattr("FOR_1_Paper.sca.sca_utils.get_hash", mock_get_hash)
    monkeypatch.setattr(
        "FOR_1_Paper.sca.sca_utils.run_or_load_model", lambda *a, **k: df_reg
    )
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa)
    monkeypatch.setattr(
        "FOR_1_Paper.sca.sca_utils.safe_save_dataframe", mock_save_dataframe
    )

    # Call function
    all_factors, all_betas, all_results = run_sca(
        regression_specs=regression_specs,
        analysis_specs=analysis_specs,
        reg_vars=object(),
        df_for=pd.DataFrame(),
        which_factor=which_factor,
        which_var=which_var,
        force_rerun=False,
    )

    # Test length of output
    assert len(all_factors) == 1
    assert len(all_betas) == 1
    assert len(all_results) == 1

    # Test whether factors and betas have the same ID index
    assert all_factors[0].index.equals(all_betas[0].index)

    # Test whether correlation between factors and betas is 1
    assert all_results[0]["effect"] == pytest.approx(1.0, rel=1e-6)

    # Test whether we save the correct output
    assert saved["name"].endswith("sca_SCAHASH.pkl")
    assert saved["data_dir"] == "for_data/sca/"
    assert saved["columns"] == [
        "model_id",
        "reg_param",
        "fa",
        "fa_param",
        "effect",
        "p_value",
    ]
    assert saved["row"] == {
        "effect": 0.9999999999999999,
        "fa": {"fa": 3},
        "fa_param": "X",
        "model_id": "model_1",
        "p_value": 0.0,
        "reg_param": 1,
    }


def test_run_sca_overlap_assert_1(monkeypatch, tmp_path):
    """Tests the function that runs the SCA with not enough overlapping subjects."""

    # Inputs
    regression_specs = {"regression_11": {"reg_param": 1}}
    analysis_specs = {"fa_11": {"fa": {"fa": 3}, "fa_param": "X"}}
    which_var = "beta_1"

    which_factor = pd.DataFrame({"fa_hash": ["FAHASH"], "max_index": ["F1"]})

    # Mock dataframes
    df_reg = make_reg_df(which_var=which_var, n=64)  # 64, not 65
    df_fa = make_df_sca_fa_with_id(factor_name="F1")

    monkeypatch.setattr("FOR_1_Paper.sca.sca_utils.get_hash", mock_get_hash)
    monkeypatch.setattr(
        "FOR_1_Paper.sca.sca_utils.run_or_load_model", lambda *a, **k: df_reg
    )
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa)
    monkeypatch.setattr(
        "FOR_1_Paper.sca.sca_utils.safe_save_dataframe", mock_save_dataframe
    )

    with pytest.raises(AssertionError, match="Not enough overlapping subjects"):
        run_sca(
            regression_specs=regression_specs,
            analysis_specs=analysis_specs,
            reg_vars=object(),
            df_for=pd.DataFrame(),
            which_factor=which_factor,
            which_var=which_var,
            force_rerun=False,
        )


def test_run_sca_overlap_assert_2(monkeypatch, tmp_path):
    """Tests the function that runs the SCA with ID mismatch."""

    # Inputs
    regression_specs = {"regression_11": {"reg_param": 1}}
    analysis_specs = {"fa_11": {"fa": {"fa": 3}, "fa_param": "X"}}
    which_var = "beta_1"

    which_factor = pd.DataFrame({"fa_hash": ["FAHASH"], "max_index": ["F1"]})

    # Mock dataframes
    df_reg = make_reg_df(which_var=which_var)
    df_fa = make_df_sca_fa_with_id(factor_name="F1")

    # Break alignment by shifting IDs
    df_fa["ID"] += 1

    monkeypatch.setattr("FOR_1_Paper.sca.sca_utils.get_hash", mock_get_hash)
    monkeypatch.setattr(
        "FOR_1_Paper.sca.sca_utils.run_or_load_model", lambda *a, **k: df_reg
    )
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa)
    monkeypatch.setattr(
        "FOR_1_Paper.sca.sca_utils.safe_save_dataframe", mock_save_dataframe
    )

    with pytest.raises(AssertionError, match="Not enough overlapping subjects"):
        run_sca(
            regression_specs=regression_specs,
            analysis_specs=analysis_specs,
            reg_vars=object(),
            df_for=pd.DataFrame(),
            which_factor=which_factor,
            which_var=which_var,
            force_rerun=False,
        )


def test_correlate_reg_fa(monkeypatch):
    """Tests the function that correlates factors and betas."""

    # Inputs
    which_var = "beta_1"
    df_reg_mock = make_reg_df(which_var=which_var, n=65)
    analysis_spec = {"fa": {"fa": "simple", "data_type": "big_data"}}
    fa_hash = mock_get_hash(analysis_spec)
    df_fa = make_df_sca_fa_with_id(factor_name="F1")
    which_factor = pd.DataFrame({"fa_hash": ["FAHASH"], "max_index": ["F1"]})

    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa)

    analysis_result, df_sca, df_reg, factor_name = correlate_reg_fa(
        which_factor, fa_hash, df_fa, df_reg_mock, which_var
    )

    assert analysis_result["effect"] == pytest.approx(1.0, rel=1e-6)
    assert df_sca.equals(df_fa)
    assert df_reg.equals(df_reg_mock)
    assert factor_name == "F1"


def test_correlate_reg_fa_assert_not_enough_subs(monkeypatch):
    """Tests the function that correlates factors and betas.

    In this case, the function should raise an AssertionError because there are not enough subjects.
    """

    # Inputs
    which_var = "beta_1"
    df_reg_mock = make_reg_df(which_var=which_var, n=64)
    analysis_spec = {"fa": {"fa": "simple", "data_type": "big_data"}}
    fa_hash = mock_get_hash(analysis_spec)
    df_fa = make_df_sca_fa_with_id(factor_name="F1")
    which_factor = pd.DataFrame({"fa_hash": ["FAHASH"], "max_index": ["F1"]})

    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa)

    with pytest.raises(AssertionError, match="Not enough overlapping subjects"):
        correlate_reg_fa(which_factor, fa_hash, df_fa, df_reg_mock, which_var)


def test_filter_subjects(monkeypatch):
    """Tests the function that filters subjects."""

    which_var = "beta_1"
    df_reg_mock = make_reg_df(which_var=which_var, n=65)
    analysis_spec = {"fa": {"fa": "simple", "data_type": "big_data"}}
    df_fa_with_id = make_df_sca_fa_with_id(factor_name="F1")

    monkeypatch.setattr("FOR_1_Paper.sca.sca_utils.get_hash", mock_get_hash)
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa_with_id)

    df_sca_fa, df_reg, fa_hash = filter_subjects(analysis_spec, df_reg_mock)

    assert df_sca_fa.equals(df_fa_with_id)
    assert df_reg.equals(df_reg_mock)
    assert fa_hash == "FAHASH"


def test_filter_subjects_with_subj_num(monkeypatch):
    """Tests the function that filters subjects.

    In this case, the function needs to use the subject number instead of ID."""

    which_var = "beta_1"
    df_reg_mock = make_reg_df(which_var=which_var, n=65)
    analysis_spec = {"fa": {"fa": "simple", "data_type": "big_data"}}
    df_fa_with_id = make_df_sca_fa_with_id(factor_name="F1")
    df_fa_with_subj_num = make_df_sca_fa_with_subj_num(factor_name="F1")

    monkeypatch.setattr("FOR_1_Paper.sca.sca_utils.get_hash", mock_get_hash)
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa_with_subj_num)

    df_sca_fa, df_reg, fa_hash = filter_subjects(analysis_spec, df_reg_mock)

    assert df_sca_fa.equals(df_fa_with_id)
    assert df_reg.equals(df_reg_mock)
    assert fa_hash == "FAHASH"


def test_filter_subjects_match_subjects(monkeypatch):
    """Tests the function that filters subjects.

    In this case, the function needs to match subjects between df_reg and df_fa."""

    which_var = "beta_1"
    df_reg_mock = make_reg_df(which_var=which_var, n=70)
    analysis_spec = {"fa": {"fa": "simple", "data_type": "big_data"}}
    df_fa_with_id = make_df_sca_fa_with_id(factor_name="F1")
    df_fa_with_subj_num = make_df_sca_fa_with_subj_num(factor_name="F1")

    monkeypatch.setattr("FOR_1_Paper.sca.sca_utils.get_hash", mock_get_hash)
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa_with_subj_num)

    df_sca_fa, df_reg, fa_hash = filter_subjects(analysis_spec, df_reg_mock)

    assert df_sca_fa.equals(df_fa_with_id)
    assert df_reg.equals(df_reg_mock[:65])
    assert fa_hash == "FAHASH"


def test_run_or_load_model_load(monkeypatch):
    """Tests the function that gets the regression model.

    In this case, we pretend the file exists, and we load the data.
    """

    # Create input variables
    reg_name = "regression_11"
    reg_spec = {"beta_0": True, "beta_1": True, "beta_2": False}
    reg_vars = type("RegVars", (), {"n_sp": 5})()  # mock object
    df_for = pd.DataFrame()

    # Define what should be "loaded" instead of running model
    mock_loaded = {"beta_1": [1, 2, 3]}

    # Patch os.path.exists so it pretends the file exists
    monkeypatch.setattr(os.path, "exists", lambda path: True)

    # Patch pickle.load so it returns mock data
    monkeypatch.setattr(pickle, "load", lambda f: mock_loaded)

    # Patch open() to a dummy context manager (we don’t use its content)
    class DummyFile:
        """Dummy file class to replace open()."""

        def __enter__(self):
            return None

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("builtins.open", lambda path, mode: DummyFile())

    df_reg = run_or_load_model(reg_name, reg_spec, reg_vars, df_for)

    # Test whether we return the loaded data
    assert df_reg == mock_loaded


def test_run_or_load_model_run(monkeypatch):
    """Tests the function that gets the regression model.

    In this case, the file does not exist, so we pretend to run the model.
    """

    # Mock function for data saving
    saved = {}

    def mock_save_dataframe_run_model(df):
        """Mock function for data saving."""

        saved["name"] = getattr(df, "name", None)
        saved["df"] = df.copy()

    # Create input variables
    reg_name = "regression_11"
    reg_spec = {"beta_0": True}
    reg_vars = RegVars()
    df_for = pd.DataFrame()

    # Define what should be the result of the regression that we "run"
    mock_result = pd.DataFrame({"beta_1": [2, 3, 4]})

    # Patch save_safe_dataframe function so we can check the saved data
    monkeypatch.setattr(
        "FOR_1_Paper.sca.sca_utils.safe_save_dataframe", mock_save_dataframe_run_model
    )

    # Define mock class to replace RegressionFor
    class MockRegression:
        def __init__(self, reg_vars):
            self.reg_vars = reg_vars

        def parallel_estimation(self, df_for, prior_columns):
            return mock_result

    # Patch RegressionFor
    monkeypatch.setattr(sca_utils, "RegressionFor", MockRegression, raising=True)
    df_reg_result = run_or_load_model(reg_name, reg_spec, reg_vars, df_for)

    # Test whether we return the loaded data
    assert saved["name"] == "regression_11_5sp_beab49c1afc5dda7242065f6e0547ef5"
    assert saved["df"].equals(mock_result)
    assert df_reg_result.equals(mock_result)


def test_fisher_z_median():
    """Tests the function that calculates the Fisher Z median."""

    rs = [0.1, 0.3, -0.1, -0.3]
    z_median = fisher_z_median(rs)
    assert z_median == pytest.approx(0.0, abs=1e-6)


def test_fisher_z_median_raises_non_finite():
    """Tests the function that calculates the Fisher Z median.

    Raises ValueError because of non-finite correlation.
    """

    rs = [0.1, 0.3, np.nan, -0.3]
    with pytest.raises(ValueError, match=r"Found 1 non-finite correlation value\(s\)"):
        fisher_z_median(rs)


# import pandas as pd
def test_run_sca_permutation_test():
    """Tests the function that runs the SCA permutation test."""

    # Create all_factors input
    all_factors = [
        pd.Series(
            [1, 2, 3, 4, 5], index=[10011, 10012, 10013, 10014, 10015], name="beta_1"
        ),
        pd.Series(
            [6, 7, 8, 9, 10], index=[10011, 10012, 10013, 10014, 10015], name="beta_1"
        ),
    ]

    # Create all_betas input
    all_betas = [
        pd.DataFrame(
            {"ID": [10011, 10012, 10013, 10014, 10015], "beta_1": [1, 2, 3, 4, 5]}
        ).set_index("ID"),
        pd.DataFrame(
            {"ID": [10011, 10012, 10013, 10014, 10015], "beta_1": [6, 7, 8, 9, 10]}
        ).set_index("ID"),
    ]

    # Compute correlations
    r_0, p_0 = stats.pearsonr(all_factors[0].values, all_betas[0].values.flatten())
    r_1, p_1 = stats.pearsonr(all_factors[1].values, all_betas[1].values.flatten())

    # Create all_results input
    all_results = [{"effect": r_0, "p_value": p_0}, {"effect": r_1, "p_value": p_1}]

    # Run permutation test
    p_value = run_sca_permutation_test(
        all_factors, all_betas, all_results, expected_n_subj=5
    )

    # Test computed p-value
    assert p_value == pytest.approx(0.021978, rel=1e-3)


def test_run_sca_permutation_test_subj_mismatch():
    """Tests the function that runs the SCA permutation test.

    Raises AssertionError because not enough overlapping subjects.
    """

    # Create all_factors input
    all_factors = [
        pd.Series(
            [1, 2, 3, 4, 5], index=[10011, 10012, 10013, 10014, 10015], name="beta_1"
        ),
        pd.Series(
            [6, 7, 8, 9, 10], index=[10011, 10012, 10013, 10014, 10015], name="beta_1"
        ),
    ]

    # Create all_betas input
    all_betas = [
        pd.DataFrame(
            {"ID": [10011, 10012, 10013, 10014, 10015], "beta_1": [1, 2, 3, 4, 5]}
        ).set_index("ID"),
        pd.DataFrame(
            {"ID": [10011, 10012, 10013, 10014, 10015], "beta_1": [6, 7, 8, 9, 10]}
        ).set_index("ID"),
    ]

    # Compute correlations
    r_0, p_0 = stats.pearsonr(all_factors[0].values, all_betas[0].values.flatten())
    r_1, p_1 = stats.pearsonr(all_factors[1].values, all_betas[1].values.flatten())

    # Create all_results input
    all_results = [{"effect": r_0, "p_value": p_0}, {"effect": r_1, "p_value": p_1}]

    with pytest.raises(AssertionError, match="Not enough overlapping subjects"):
        run_sca_permutation_test(all_factors, all_betas, all_results)


def test_run_sca_permutation_test_r_mismatch():
    """Tests the function that runs the SCA permutation test.

    Raises AssertionError because correlation mismatch.
    """

    # Create all_factors input
    all_factors = [
        pd.Series(
            [1, 2, 3, 4, 5], index=[10011, 10012, 10013, 10014, 10015], name="beta_1"
        ),
        pd.Series(
            [6, 7, 8, 9, 10], index=[10011, 10012, 10013, 10014, 10015], name="beta_1"
        ),
    ]

    # Create all_betas input
    all_betas = [
        pd.DataFrame(
            {"ID": [10011, 10012, 10013, 10014, 10015], "beta_1": [1, 2, 3, 4, 5]}
        ).set_index("ID"),
        pd.DataFrame(
            {"ID": [10011, 10012, 10013, 10014, 10015], "beta_1": [6, 7, 8, 9, 10]}
        ).set_index("ID"),
    ]

    # Compute correlations
    r_0, p_0 = stats.pearsonr(all_factors[0].values, all_betas[0].values.flatten())
    r_1, p_1 = stats.pearsonr(all_factors[1].values, all_betas[1].values.flatten())

    # Create all_results input
    all_results = [{"effect": 0.5, "p_value": p_0}, {"effect": r_1, "p_value": p_1}]

    with pytest.raises(AssertionError, match="Correlation mismatch"):
        run_sca_permutation_test(all_factors, all_betas, all_results, expected_n_subj=5)


def test_perm_pval_not_significant():
    """Tests the function that calculates the permutation p-value.

    In this case, the result is not significant.
    """

    z_value_obs = 0.5
    z_values_perm = np.array([0.3, 0.4, 0.5, -0.6, -0.7])
    p_value = perm_pval(z_value_obs, z_values_perm)
    assert p_value == pytest.approx(0.3333, rel=1e-3)


def test_perm_pval_not_significant_plus():
    """Tests the function that calculates the permutation p-value.

    In this case, the result is not significant, and we run a two-sided test.
    """

    z_value_obs = 0.5
    z_values_perm = np.array([0.3, 0.4, 0.5, -0.6, -0.7])
    p_value = perm_pval(z_value_obs, z_values_perm, tail="two-sided")
    assert p_value == pytest.approx(0.6666, rel=1e-3)


def test_perm_pval_not_significant_minus():
    """Tests the function that calculates the permutation p-value.

    In this case, the result is not significant, and we run a one-sided (negative) test.
    """

    z_value_obs = 0.5
    z_values_perm = np.array([0.3, 0.4, 0.5, -0.6, -0.7])
    p_value = perm_pval(z_value_obs, z_values_perm, tail="-")
    assert p_value == pytest.approx(1, rel=1e-3)


def test_perm_pval_significant():
    """Tests the function that calculates the permutation p-value.

    In this case, the result is significant. One-sided (positive) test.
    """

    z_value_obs = 0.999
    z_values_perm = np.repeat([0.1, 0.2, 0.3, 0.4, 0.5], 10)
    p_value = perm_pval(z_value_obs, z_values_perm)
    assert p_value == pytest.approx(1 / 51, rel=1e-3)


def test_perm_pval_value_error_obs():
    """Tests the function that calculates the permutation p-value.

    Raises a ValueError that the observed z-value is not finite.
    """

    z_value_obs = np.nan
    z_values_perm = np.repeat([0.1, 0.2, 0.3, 0.4, 0.5], 10)
    with pytest.raises(ValueError, match="Observed z-value must be finite."):
        perm_pval(z_value_obs, z_values_perm)


def test_perm_pval_value_error_perm():
    """Tests the function that calculates the permutation p-value.

    Raises a ValueError that the permuted z-values are not finite.
    """

    z_value_obs = 0.5
    z_values_perm = np.repeat([0.1, np.nan, 0.3, 0.4, 0.5], 10)
    with pytest.raises(ValueError, match="Permuted z-values must be finite."):
        perm_pval(z_value_obs, z_values_perm)
