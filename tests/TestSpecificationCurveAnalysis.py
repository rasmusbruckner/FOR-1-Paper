"""Unit and integration tests for SpecificationCurveAnalysis.py"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
from rbmpy import AgentVars

from FOR_1_Paper.modeling.ForEstVars import ForEstVars
from FOR_1_Paper.regression.ForRegVars import RegVars
from FOR_1_Paper.sca.SpecificationCurveAnalysis import (
    SpecificationCurveAnalysis, check_fa_spec, fa_candidates, fisher_z_median,
    get_hash, perm_pval, run_or_load_regression, run_or_load_rbm)


def test_init():
    """Tests the initialization of the SCA object."""

    sca = SpecificationCurveAnalysis()
    assert sca.n_subj is None
    assert sca.pool is None
    assert sca.sca_fa_filetype == ".pkl"
    assert sca.which_analysis_str == "fixed_LR"
    assert sca.var_names is None
    assert sca.var_rules is None
    assert sca.expected_n_subj == 65
    assert sca.significance_counter is None
    assert sca.counter is None
    assert sca.all_quest_data is None
    assert sca.all_parameters is None
    assert sca.all_results is None


def test_run_sca(monkeypatch, tmp_path):
    """Tests the function that runs the SCA for the sum scores."""

    # Inputs
    regression_specs = {
        "regression_11": {
            "regression_11": {"beta_1": 1},
            "dependent_variable": {"beta_1": 1},
        }
    }

    rbm_specs = {"rbm_11": {"h": True}}

    which_var_quest = {"sumscore"}

    # Mock dataframes
    df_reg = make_analysis_df()
    df_questionnaires = make_df_sca_quest_with_id(var_name="sumscore")
    df_rbm = make_analysis_df(which_var="h")

    # Mock out get_hash
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )

    # Mock out run_or_load_regression
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.run_or_load_regression",
        lambda *a, **k: df_reg,
    )

    # Mock out read_pickle
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_questionnaires)

    # Mock out safe_save_dataframe
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.safe_save_dataframe",
        mock_save_dataframe,
    )

    # Mock out run_or_load_rbm
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.run_or_load_rbm",
        lambda *a, **k: df_rbm,
    )

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    # Call function
    sca.run_sca(
        regression_specs=regression_specs,
        reg_vars=RegVars(),
        rbm_specs=rbm_specs,
        est_vars=ForEstVars(),
        agent_vars=AgentVars(),
        df_for=pd.DataFrame(),
        df_rbm=pd.DataFrame(),
        df_questionnaires=df_questionnaires,
        which_var_quest=which_var_quest,
    )

    # Test length of output
    assert len(sca.all_quest_data) == 2
    assert len(sca.all_parameters) == 2
    assert len(sca.all_results) == 2

    # Test whether questionnaire and betas have the same ID index
    assert sca.all_quest_data[0].index.equals(sca.all_parameters[0].index)

    # Test whether correlation between questionnaires and betas is 1
    assert sca.all_results[0]["effect"] == pytest.approx(1.0, rel=1e-6)

    # Test whether we save the correct output for regression part
    assert saved["sca_REGHASH"]["name"].endswith("sca_REGHASH")
    assert saved["sca_REGHASH"]["data_dir"] == "for_data/sca_temp/"
    assert saved["sca_REGHASH"]["columns"] == [
        "model_id",
        "dependent_variable",
        "quest_variable",
        "beta_1",
        "effect",
        "p_value",
        "dv",
        "qv",
        "bic"
    ]
    assert saved["sca_REGHASH"]["row"] == {
        "model_id": "model_1",
        "dependent_variable": "beta_1",
        "quest_variable": "sumscore",
        "beta_1": True,
        "bic": -325.0,
        "effect": 1.0,
        "p_value": 0.0,
        "dv": "beta_1",
        "qv": "sumscore"
    }

    # Test whether we save the correct output for RBM part
    assert saved["sca_RBMHASH"]["name"].endswith("sca_RBMHASH")
    assert saved["sca_RBMHASH"]["data_dir"] == "for_data/sca_temp/"
    assert saved["sca_RBMHASH"]["columns"] == [
        "model_id",
        "dependent_variable",
        "quest_variable",
        "h",
        "effect",
        "p_value",
        "dv",
        "qv",
        "bic"
    ]

    assert saved["sca_RBMHASH"]["row"] == {
        "model_id": "model_2",
        "bic": -325.0,
        "dependent_variable": "h",
        "quest_variable": "sumscore",
        "h": True,
        "effect": 1.0,
        "p_value": 0.0,
        "dv": "h",
        "qv": "sumscore",
    }


def test_run_sca_flip_beta4(monkeypatch, tmp_path):
    """Tests the function that runs the SCA for the sum scores."""

    # Inputs
    regression_specs = {
        "regression_11": {
            "regression_11": {"beta_4": 1},
            "dependent_variable": {"beta_4": 1},
        }
    }

    rbm_specs = {"rbm_11": {"h": True}}

    which_var_quest = {"sumscore"}

    # Mock dataframes
    df_reg = make_analysis_df(which_var="beta_4")
    df_questionnaires = make_df_sca_quest_with_id(var_name="sumscore")
    df_rbm = make_analysis_df(which_var="h")

    # Mock out get_hash
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )

    # Mock out run_or_load_regression
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.run_or_load_regression",
        lambda *a, **k: df_reg,
    )

    # Mock out read_pickle
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_questionnaires)

    # Mock out safe_save_dataframe
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.safe_save_dataframe",
        mock_save_dataframe,
    )

    # Mock out run_or_load_rbm
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.run_or_load_rbm",
        lambda *a, **k: df_rbm,
    )

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    # Call function
    sca.run_sca(
        regression_specs=regression_specs,
        reg_vars=RegVars(),
        rbm_specs=rbm_specs,
        est_vars=ForEstVars(),
        agent_vars=AgentVars(),
        df_for=pd.DataFrame(),
        df_rbm=pd.DataFrame(),
        df_questionnaires=df_questionnaires,
        which_var_quest=which_var_quest,
    )

    # Test length of output
    assert len(sca.all_quest_data) == 2
    assert len(sca.all_parameters) == 2
    assert len(sca.all_results) == 2

    # Test whether questionnaire and betas have the same ID index
    assert sca.all_quest_data[0].index.equals(sca.all_parameters[0].index)

    # Test whether correlation between questionnaires and betas is 1
    assert sca.all_results[0]["effect"] == pytest.approx(-1.0, rel=1e-6)

    # Test whether we save the correct output for regression part
    assert saved["sca_REGHASH"]["name"].endswith("sca_REGHASH")
    assert saved["sca_REGHASH"]["data_dir"] == "for_data/sca_temp/"
    assert saved["sca_REGHASH"]["columns"] == [
        "model_id",
        "dependent_variable",
        "quest_variable",
        "beta_4",
        "effect",
        "p_value",
        "dv",
        "qv",
        "bic"
    ]
    assert saved["sca_REGHASH"]["row"] == {
        "model_id": "model_1",
        "dependent_variable": "beta_4",
        "quest_variable": "sumscore",
        "beta_4": True,
        "bic": -325.0,
        "effect": -1.0,
        "p_value": 0.0,
        "dv": "beta_4",
        "qv": "sumscore",
    }

    # Test whether we save the correct output for RBM part
    assert saved["sca_RBMHASH"]["name"].endswith("sca_RBMHASH")
    assert saved["sca_RBMHASH"]["data_dir"] == "for_data/sca_temp/"
    assert saved["sca_RBMHASH"]["columns"] == [
        "model_id",
        "dependent_variable",
        "quest_variable",
        "h",
        "effect",
        "p_value",
        "dv",
        "qv",
        "bic"
    ]

    assert saved["sca_RBMHASH"]["row"] == {
        "model_id": "model_2",
        "bic": -325.0,
        "dependent_variable": "h",
        "quest_variable": "sumscore",
        "h": True,
        "effect": 1.0,
        "p_value": 0.0,
        "dv": "h",
        "qv": "sumscore",
    }


def test_compute_sca_correlation(monkeypatch, tmp_path):
    """Tests the function that computes the SCA correlation."""

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    reg_spec = {"regression_11": {"beta_1": 1}, "dependent_variable": {"beta_1": 1}}

    reg_name = "regression_11"
    dep_name = "beta_1"
    quest = "sumscore"

    df_questionnaires = make_df_sca_quest_with_id(var_name="sumscore")
    df_reg = make_analysis_df()

    # Initialize counter for significant results
    sca.significance_counter = 0
    sca.counter = 1

    # Initialize output lists
    sca.all_quest_data = []
    sca.all_parameters = []
    sca.all_results = []

    # Mock out safe_save_dataframe
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.safe_save_dataframe",
        mock_save_dataframe,
    )

    # Mock out get_hash
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )

    sca.compute_sca_correlations(
        df_questionnaires,
        df_reg,
        dep_name,
        quest,
        reg_spec[reg_name],
    )

    # Test length of output
    assert len(sca.all_quest_data) == 1
    assert len(sca.all_parameters) == 1
    assert len(sca.all_results) == 1

    # Test whether questionnaire and betas have the same ID index
    assert sca.all_quest_data[0].index.equals(sca.all_parameters[0].index)

    # Test whether correlation between questionnaires and betas is 1
    assert sca.all_results[0]["effect"] == pytest.approx(1.0, rel=1e-6)

    # Test whether we save the correct output for regression part
    assert saved["sca_REGHASH"]["name"].endswith("sca_REGHASH")
    assert saved["sca_REGHASH"]["data_dir"] == "for_data/sca_temp/"
    assert saved["sca_REGHASH"]["columns"] == [
        "model_id",
        "dependent_variable",
        "quest_variable",
        "beta_1",
        "effect",
        "p_value",
        "dv",
        "qv",
        "bic"
    ]

    assert saved["sca_REGHASH"]["row"] == {
        "model_id": "model_1",
        "dependent_variable": "beta_1",
        "quest_variable": "sumscore",
        "beta_1": True,
        "bic": -325.0,
        "effect": 1.0,
        "p_value": 0.0,
        "dv": "beta_1",
        "qv": "sumscore",
    }


def test_compute_sca_correlation_with_subj_num(monkeypatch, tmp_path):
    """Tests the function that computes the SCA correlation."""

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    reg_spec = {"regression_11": {"beta_1": 1}, "dependent_variable": {"beta_1": 1}}

    reg_name = "regression_11"
    dep_name = "beta_1"
    quest = "sumscore"

    df_questionnaires = make_df_sca_quest_with_subj_num(var_name="sumscore")
    df_reg = make_analysis_df()

    # Initialize counter for significant results
    sca.significance_counter = 0
    sca.counter = 1

    # Initialize output lists
    sca.all_quest_data = []
    sca.all_parameters = []
    sca.all_results = []

    # Mock out safe_save_dataframe
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.safe_save_dataframe",
        mock_save_dataframe,
    )

    # Mock out get_hash
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )

    sca.compute_sca_correlations(
        df_questionnaires,
        df_reg,
        dep_name,
        quest,
        reg_spec[reg_name],
    )

    # Test length of output
    assert len(sca.all_quest_data) == 1
    assert len(sca.all_parameters) == 1
    assert len(sca.all_results) == 1

    # Test whether questionnaire and betas have the same ID index
    assert sca.all_quest_data[0].index.equals(sca.all_parameters[0].index)

    # Test whether correlation between questionnaires and betas is 1
    assert sca.all_results[0]["effect"] == pytest.approx(1.0, rel=1e-6)

    # Test whether we save the correct output for regression part
    assert saved["sca_REGHASH"]["name"].endswith("sca_REGHASH")
    assert saved["sca_REGHASH"]["data_dir"] == "for_data/sca_temp/"
    assert saved["sca_REGHASH"]["columns"] == [
        "model_id",
        "dependent_variable",
        "quest_variable",
        "beta_1",
        "effect",
        "p_value",
        "dv",
        "qv",
        "bic"
    ]

    assert saved["sca_REGHASH"]["row"] == {
        "model_id": "model_1",
        "dependent_variable": "beta_1",
        "quest_variable": "sumscore",
        "beta_1": True,
        "bic": -325.0,
        "effect": 1.0,
        "p_value": 0.0,
        "dv": "beta_1",
        "qv": "sumscore",
    }


def test_compute_sca_correlation_assert_1(monkeypatch, tmp_path):
    """Tests the function that computes the SCA correlation with not enough overlapping subjects."""

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    reg_spec = {"regression_11": {"beta_1": 1}, "dependent_variable": {"beta_1": 1}}

    reg_name = "regression_11"
    dep_name = "beta_1"
    quest = "sumscore"

    df_questionnaires = make_df_sca_quest_with_id(var_name="sumscore")
    df_reg = make_analysis_df(n=64)  # 64, not 65

    # Initialize counter for significant results
    sca.significance_counter = 0
    sca.counter = 1

    # Initialize output lists
    sca.all_quest_data = []
    sca.all_parameters = []
    sca.all_results = []

    # Mock out safe_save_dataframe
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.safe_save_dataframe",
        mock_save_dataframe,
    )

    # Mock out get_hash
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )

    with pytest.raises(AssertionError, match="Not enough overlapping subjects"):
        sca.compute_sca_correlations(
            df_questionnaires,
            df_reg,
            dep_name,
            quest,
            reg_spec[reg_name],
        )


def test_compute_sca_correlation_assert_2(monkeypatch, tmp_path):
    """Tests the function that computes the SCA correlation with not enough overlapping subjects."""

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    reg_spec = {"regression_11": {"beta_1": 1}, "dependent_variable": {"beta_1": 1}}

    reg_name = "regression_11"
    dep_name = "beta_1"
    quest = "sumscore"

    df_questionnaires = make_df_sca_quest_with_id(var_name="sumscore")
    df_reg = make_analysis_df(n=65)

    # Break alignment by shifting IDs
    df_questionnaires["ID"] += 1

    # Initialize counter for significant results
    sca.significance_counter = 0
    sca.counter = 1

    # Initialize output lists
    sca.all_quest_data = []
    sca.all_parameters = []
    sca.all_results = []

    # Mock out safe_save_dataframe
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.safe_save_dataframe",
        mock_save_dataframe,
    )

    # Mock out get_hash
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )

    with pytest.raises(AssertionError, match="Not enough overlapping subjects"):
        sca.compute_sca_correlations(
            df_questionnaires,
            df_reg,
            dep_name,
            quest,
            reg_spec[reg_name],
        )


def test_run_sca_fa(monkeypatch, tmp_path):
    """Tests the function that runs the SCA."""

    # Inputs
    regression_specs = {"regression_11": {"reg_param": 1}}
    analysis_specs = {"fa_11": {"fa": {"fa": 3}, "fa_param": "X"}}
    which_var = "beta_1"

    which_factor = pd.DataFrame({"fa_hash": ["FAHASH"], "max_index": ["F1"]})

    # Mock dataframes
    df_reg = make_analysis_df(which_var=which_var)
    df_fa = make_df_sca_quest_with_id(var_name="F1")

    # Mock out get_hash
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )

    # Mock out run_or_load_regression
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.run_or_load_regression",
        lambda *a, **k: df_reg,
    )

    # Mock out read_pickle
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa)

    # Mock out safe_save_dataframe
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.safe_save_dataframe",
        mock_save_dataframe,
    )

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    # Call function
    all_factors, all_parameters, all_results = sca.run_sca_fa(
        regression_specs=regression_specs,
        analysis_specs=analysis_specs,
        reg_vars=RegVars(),
        df_for=pd.DataFrame(),
        which_factor=which_factor,
        which_var=which_var,
        force_rerun=False,
    )

    # Test length of output
    assert len(all_factors) == 1
    assert len(all_parameters) == 1
    assert len(all_results) == 1

    # Test whether factors and betas have the same ID index
    assert all_factors[0].index.equals(all_parameters[0].index)

    # Test whether correlation between factors and betas is 1
    assert all_results[0]["effect"] == pytest.approx(1.0, rel=1e-6)

    # Test whether we save the correct output
    assert saved["sca_REGHASH"]["name"].endswith("sca_REGHASH")
    assert saved["sca_REGHASH"]["data_dir"] == "for_data/sca_temp/"
    assert saved["sca_REGHASH"]["columns"] == [
        "model_id",
        "reg_param",
        "fa",
        "fa_param",
        "effect",
        "p_value",
    ]
    assert saved["sca_REGHASH"]["row"] == {
        "model_id": "model_1",
        "reg_param": 1,
        "fa": {"fa": 3},
        "fa_param": "X",
        "effect": 1.0,
        "p_value": 0.0,
    }


def test_run_sca_fa_with_subj_num(monkeypatch, tmp_path):
    """Tests the function that runs the SCA with subj_num input."""

    # Inputs
    regression_specs = {"regression_11": {"reg_param": 1}}
    analysis_specs = {"fa_11": {"fa": {"fa": 3}, "fa_param": "X"}}
    which_var = "beta_1"

    which_factor = pd.DataFrame({"fa_hash": ["FAHASH"], "max_index": ["F1"]})

    # Mock dataframes
    df_reg = make_analysis_df(which_var=which_var)
    df_fa = make_df_sca_quest_with_subj_num(var_name="F1")

    # Mock out get_hash
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )

    # Mock out run_or_load_regression
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.run_or_load_regression",
        lambda *a, **k: df_reg,
    )

    # Mock out read_pickle
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa)

    # Mock out safe_save_dataframe
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.safe_save_dataframe",
        mock_save_dataframe,
    )

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    # Call function
    all_factors, all_parameters, all_results = sca.run_sca_fa(
        regression_specs=regression_specs,
        analysis_specs=analysis_specs,
        reg_vars=RegVars(),
        df_for=pd.DataFrame(),
        which_factor=which_factor,
        which_var=which_var,
        force_rerun=False,
    )

    # Test length of output
    assert len(all_factors) == 1
    assert len(all_parameters) == 1
    assert len(all_results) == 1

    # Test whether factors and betas have the same ID index
    assert all_factors[0].index.equals(all_parameters[0].index)

    # Test whether correlation between factors and betas is 1
    assert all_results[0]["effect"] == pytest.approx(1.0, rel=1e-6)

    # Test whether we save the correct output
    assert saved["sca_REGHASH"]["name"].endswith("sca_REGHASH")
    assert saved["sca_REGHASH"]["data_dir"] == "for_data/sca_temp/"
    assert saved["sca_REGHASH"]["columns"] == [
        "model_id",
        "reg_param",
        "fa",
        "fa_param",
        "effect",
        "p_value",
    ]
    assert saved["sca_REGHASH"]["row"] == {
        "model_id": "model_1",
        "reg_param": 1,
        "fa": {"fa": 3},
        "fa_param": "X",
        "effect": 1.0,
        "p_value": 0.0,
    }


def test_run_sca_fa_overlap_assert_1(monkeypatch, tmp_path):
    """Tests the function that runs the SCA with not enough overlapping subjects."""

    # Inputs
    regression_specs = {"regression_11": {"reg_param": 1}}
    analysis_specs = {"fa_11": {"fa": {"fa": 3}, "fa_param": "X"}}
    which_var = "beta_1"

    which_factor = pd.DataFrame({"fa_hash": ["FAHASH"], "max_index": ["F1"]})

    # Mock dataframes
    df_reg = make_analysis_df(which_var=which_var, n=64)  # 64, not 65
    df_fa = make_df_sca_quest_with_id(var_name="F1")

    # Mock out get_hash
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )

    # Mock out run_or_load_regression
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.run_or_load_regression",
        lambda *a, **k: df_reg,
    )
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa)
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.safe_save_dataframe",
        mock_save_dataframe,
    )

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    with pytest.raises(AssertionError, match="Not enough overlapping subjects"):
        sca.run_sca_fa(
            regression_specs=regression_specs,
            analysis_specs=analysis_specs,
            reg_vars=RegVars(),
            df_for=pd.DataFrame(),
            which_factor=which_factor,
            which_var=which_var,
            force_rerun=False,
        )


def test_run_sca_fa_overlap_assert_2(monkeypatch, tmp_path):
    """Tests the function that runs the SCA with ID mismatch."""

    # Inputs
    regression_specs = {"regression_11": {"reg_param": 1}}
    analysis_specs = {"fa_11": {"fa": {"fa": 3}, "fa_param": "X"}}
    which_var = "beta_1"

    which_factor = pd.DataFrame({"fa_hash": ["FAHASH"], "max_index": ["F1"]})

    # Mock dataframes
    df_reg = make_analysis_df(which_var=which_var)
    df_fa = make_df_sca_quest_with_id(var_name="F1")

    # Break alignment by shifting IDs
    df_fa["ID"] += 1

    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.run_or_load_regression",
        lambda *a, **k: df_reg,
    )
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa)
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.safe_save_dataframe",
        mock_save_dataframe,
    )

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    with pytest.raises(AssertionError, match="Not enough overlapping subjects"):
        sca.run_sca_fa(
            regression_specs=regression_specs,
            analysis_specs=analysis_specs,
            reg_vars=RegVars(),
            df_for=pd.DataFrame(),
            which_factor=which_factor,
            which_var=which_var,
            force_rerun=False,
        )


def test_filter_subjects(monkeypatch):
    """Tests the function that filters subjects."""

    which_var = "beta_1"
    df_reg_mock = make_analysis_df(which_var=which_var, n=65)
    analysis_spec = {"fa": {"fa": "simple", "data_type": "big_data"}}
    df_fa_with_id = make_df_sca_quest_with_id(var_name="F1")

    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa_with_id)

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    df_sca_fa, df_reg, fa_hash = sca.filter_subjects(analysis_spec, df_reg_mock)

    assert df_sca_fa.equals(df_fa_with_id)
    assert df_reg.equals(df_reg_mock)
    assert fa_hash == "FAHASH"


def test_filter_subjects_with_subj_num(monkeypatch):
    """Tests the function that filters subjects.

    In this case, the function needs to use the subject number instead of ID."""

    which_var = "beta_1"
    df_reg_mock = make_analysis_df(which_var=which_var, n=65)
    analysis_spec = {"fa": {"fa": "simple", "data_type": "big_data"}}
    df_fa_with_id = make_df_sca_quest_with_id(var_name="F1")
    df_fa_with_subj_num = make_df_sca_quest_with_subj_num(var_name="F1")

    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa_with_subj_num)

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    df_sca_fa, df_reg, fa_hash = sca.filter_subjects(analysis_spec, df_reg_mock)

    assert df_sca_fa.equals(df_fa_with_id)
    assert df_reg.equals(df_reg_mock)
    assert fa_hash == "FAHASH"


def test_filter_subjects_match_subjects(monkeypatch):
    """Tests the function that filters subjects.

    In this case, the function needs to match subjects between df_reg and df_fa."""

    which_var = "beta_1"
    df_reg_mock = make_analysis_df(which_var=which_var, n=70)
    analysis_spec = {"fa": {"fa": "simple", "data_type": "big_data"}}
    df_fa_with_id = make_df_sca_quest_with_id(var_name="F1")
    df_fa_with_subj_num = make_df_sca_quest_with_subj_num(var_name="F1")

    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.get_hash", mock_get_hash
    )
    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa_with_subj_num)

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    df_sca_fa, df_reg, fa_hash = sca.filter_subjects(analysis_spec, df_reg_mock)

    assert df_sca_fa.equals(df_fa_with_id)
    assert df_reg.equals(df_reg_mock[:65])
    assert fa_hash == "FAHASH"


def test_correlate_reg_fa(monkeypatch):
    """Tests the function that correlates factors and betas."""

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    # Inputs
    which_var = "beta_1"
    df_reg_mock = make_analysis_df(which_var=which_var, n=65)
    analysis_spec = {"fa": {"fa": "simple", "data_type": "big_data"}}
    fa_hash = mock_get_hash(analysis_spec)
    df_fa = make_df_sca_quest_with_id(var_name="F1")
    which_factor = pd.DataFrame({"fa_hash": ["FAHASH"], "max_index": ["F1"]})

    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa)

    analysis_result, df_sca, df_reg, factor_name = sca.correlate_reg_fa(
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

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    # Inputs
    which_var = "beta_1"
    df_reg_mock = make_analysis_df(which_var=which_var, n=64)
    analysis_spec = {"fa": {"fa": "simple", "data_type": "big_data"}}
    fa_hash = mock_get_hash(analysis_spec)
    df_fa = make_df_sca_quest_with_id(var_name="F1")
    which_factor = pd.DataFrame({"fa_hash": ["FAHASH"], "max_index": ["F1"]})

    monkeypatch.setattr("pandas.read_pickle", lambda path: df_fa)

    with pytest.raises(AssertionError, match="Not enough overlapping subjects"):
        sca.correlate_reg_fa(which_factor, fa_hash, df_fa, df_reg_mock, which_var)


def test_run_permutation_test():
    """Tests the function that runs the SCA permutation test."""

    # Create all_quest_data input
    all_quest_data = [
        pd.Series(
            [1, 2, 3, 4, 5], index=[10011, 10012, 10013, 10014, 10015], name="beta_1"
        ),
        pd.Series(
            [6, 7, 8, 9, 10], index=[10011, 10012, 10013, 10014, 10015], name="beta_1"
        ),
    ]

    # Create all_parameters input
    all_parameters = [
        pd.DataFrame(
            {"ID": [10011, 10012, 10013, 10014, 10015], "beta_1": [1, 2, 3, 4, 5]}
        ).set_index("ID"),
        pd.DataFrame(
            {"ID": [10011, 10012, 10013, 10014, 10015], "beta_1": [6, 7, 8, 9, 10]}
        ).set_index("ID"),
    ]

    # Compute correlations
    r_0, p_0 = stats.pearsonr(
        all_quest_data[0].values, all_parameters[0].values.flatten()
    )
    r_1, p_1 = stats.pearsonr(
        all_quest_data[1].values, all_parameters[1].values.flatten()
    )

    # Create all_results input
    all_results = [{"effect": r_0, "p_value": p_0}, {"effect": r_1, "p_value": p_1}]

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()
    sca.expected_n_subj = 5
    sca.all_quest_data = all_quest_data
    sca.all_parameters = all_parameters
    sca.all_results = all_results

    # Run permutation test
    p_value = sca.run_permutation_test()

    # Test computed p-value
    assert p_value == pytest.approx(0.021978, rel=1e-3)


def test_run_permutation_test_subj_mismatch():
    """Tests the function that runs the SCA permutation test.

    Raises AssertionError because not enough overlapping subjects.
    """

    # Create all_quest_data input
    all_quest_data = [
        pd.Series(
            [1, 2, 3, 4, 5], index=[10011, 10012, 10013, 10014, 10015], name="beta_1"
        ),
        pd.Series(
            [6, 7, 8, 9, 10], index=[10011, 10012, 10013, 10014, 10015], name="beta_1"
        ),
    ]

    # Create all_parameters input
    all_parameters = [
        pd.DataFrame(
            {"ID": [10011, 10012, 10013, 10014, 10015], "beta_1": [1, 2, 3, 4, 5]}
        ).set_index("ID"),
        pd.DataFrame(
            {"ID": [10011, 10012, 10013, 10014, 10015], "beta_1": [6, 7, 8, 9, 10]}
        ).set_index("ID"),
    ]

    # Compute correlations
    r_0, p_0 = stats.pearsonr(
        all_quest_data[0].values, all_parameters[0].values.flatten()
    )
    r_1, p_1 = stats.pearsonr(
        all_quest_data[1].values, all_parameters[1].values.flatten()
    )

    # Create all_results input
    all_results = [{"effect": r_0, "p_value": p_0}, {"effect": r_1, "p_value": p_1}]

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()
    sca.all_quest_data = all_quest_data
    sca.all_parameters = all_parameters
    sca.all_results = all_results

    with pytest.raises(AssertionError, match="Not enough overlapping subjects"):
        sca.run_permutation_test()


def test_run_permutation_test_r_mismatch():
    """Tests the function that runs the SCA permutation test.

    Raises AssertionError because correlation mismatch.
    """

    # Create all_quest_data input
    all_quest_data = [
        pd.Series(
            [1, 2, 3, 4, 5], index=[10011, 10012, 10013, 10014, 10015], name="beta_1"
        ),
        pd.Series(
            [6, 7, 8, 9, 10], index=[10011, 10012, 10013, 10014, 10015], name="beta_1"
        ),
    ]

    # Create all_parameters input
    all_parameters = [
        pd.DataFrame(
            {"ID": [10011, 10012, 10013, 10014, 10015], "beta_1": [1, 2, 3, 4, 5]}
        ).set_index("ID"),
        pd.DataFrame(
            {"ID": [10011, 10012, 10013, 10014, 10015], "beta_1": [6, 7, 8, 9, 10]}
        ).set_index("ID"),
    ]

    # Compute correlations
    r_0, p_0 = stats.pearsonr(
        all_quest_data[0].values, all_parameters[0].values.flatten()
    )
    r_1, p_1 = stats.pearsonr(
        all_quest_data[1].values, all_parameters[1].values.flatten()
    )

    # Create all_results input
    all_results = [{"effect": 0.5, "p_value": p_0}, {"effect": r_1, "p_value": p_1}]

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()
    sca.all_quest_data = all_quest_data
    sca.all_parameters = all_parameters
    sca.all_results = all_results
    sca.expected_n_subj = 5

    with pytest.raises(AssertionError, match="Correlation mismatch"):
        sca.run_permutation_test()


def test_check_fa_pool_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Tests the case that everything is complete."""

    # Monkeypatch Path to return tmp_path when instantiated with "for_data"
    def mock_path(path_str):
        """Mock function for Path."""

        if path_str == "for_data":
            return tmp_path
        return Path(path_str)

    monkeypatch.setattr("FOR_1_Paper.sca.SpecificationCurveAnalysis.Path", mock_path)

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    # Specification pool
    fa1 = {"factor_method": "ml", "rotation": "varimax", "n_factors": "3"}
    fa2 = {"factor_method": "minres", "rotation": "oblimin", "n_factors": "4"}

    # Add specification pool to sca object instance
    sca.pool = [fa1, fa2]

    # Cycle over fas to create temporary files for testing
    for fa in (fa1, fa2):

        # Create temporary file for testing
        _touch(tmp_path, _fname(get_hash(fa)))

    # Run to see if it does not raise error
    sca.check_fa_pool(all_expected=False)


def test_check_fa_pool_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Tests the case that file is not found."""

    # Monkeypatch Path to return tmp_path when instantiated with "for_data"
    def mock_path(path_str):
        """Mock function for Path."""

        if path_str == "for_data":
            return tmp_path
        return Path(path_str)

    monkeypatch.setattr("FOR_1_Paper.sca.SpecificationCurveAnalysis.Path", mock_path)

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    # Specification pool
    fa = {"factor_method": "ml", "rotation": "varimax", "n_factors": "3"}

    # Add specification pool to sca object instance
    sca.pool = [fa]

    # We haven't created a file, so this should raise an error
    with pytest.raises(FileNotFoundError):
        sca.check_fa_pool(all_expected=False)


def test_check_fa_pool_extras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Tests the case that all expected files are present, but there are extras."""

    # Monkeypatch Path to return tmp_path when instantiated with "for_data"
    def mock_path(path_str):
        """Mock function for Path."""

        if path_str == "for_data":
            return tmp_path
        return Path(path_str)

    monkeypatch.setattr("FOR_1_Paper.sca.SpecificationCurveAnalysis.Path", mock_path)

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    # Specification pool
    fa = {"factor_method": "ml", "rotation": "varimax", "n_factors": "3"}

    # Add specification pool to sca object instance
    sca.pool = [fa]

    # Create a file that the function expects
    _touch(tmp_path, _fname(get_hash(fa)))

    # Create a file that is not expected
    _touch(tmp_path, _fname("0" * 32))

    # Should raise an error because there are extras
    with pytest.raises(FileNotFoundError):
        sca.check_fa_pool(all_expected=True)


def test_build_specs_with_vars_all_included():
    """Tests the case where all variable rules apply."""

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    # Parameter space of factor analysis
    param_space = {
        "analysis_type": ["simple", "bifactor"],
        "data_type": ["big_data"],
    }

    # Analysis specifications based on our parameter space
    pool = list(fa_candidates(param_space))

    # Add specification pool to sca object instance
    sca.pool = pool

    var_rule_ids = ["psychosis"]

    var_rules = {
        "psychosis": [
            lambda fa: True,
        ],
    }

    # Add rule and rule name to SCA object instance
    sca.var_names = var_rule_ids
    sca.var_rules = var_rules

    analysis_specs = sca.build_specs_with_vars()

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

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    # Parameter space of factor analysis
    param_space = {
        "analysis_type": ["simple", "bifactor"],
        "data_type": ["big_data"],
    }

    # Analysis specifications based on our parameter space
    pool = list(fa_candidates(param_space))

    # Add specification pool to sca object instance
    sca.pool = pool

    var_rule_ids = ["g_only"]

    # Only applies to bifactor
    var_rules = {
        "g_only": [lambda fa: fa["analysis_type"] == "bifactor"],
    }

    # Add rule and rule name to SCA object instance
    sca.var_names = var_rule_ids
    sca.var_rules = var_rules

    analysis_specs = sca.build_specs_with_vars()

    expected_specs = {
        "spec_g_only_7d367668b66d4db14869a79a535b8f63": {
            "fa": {"analysis_type": "bifactor", "data_type": "big_data"},
            "g_only": True,
        }
    }

    assert analysis_specs == expected_specs


def test_passes_variable_rules_all_passed():
    """Tests the case where all variable rules apply."""

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    fa = {"analysis_type": "simple", "rotation": "oblimin"}
    var = "psychosis"

    # Always applies
    var_rules = {
        "psychosis": [
            lambda fa: True,
        ],
    }

    # Add rule to SCA object instance
    sca.var_rules = var_rules

    passed = sca.passes_variable_rules(fa, var)
    assert passed


def test_passes_variable_rules_specific_passed():
    """Tests the case where a specific variable rule applies."""

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    fa = {"analysis_type": "bifactor", "rotation": "oblimin"}
    var = "g_only"

    # Only applies to bifactor
    var_rules = {
        "g_only": [lambda fa: fa["analysis_type"] == "bifactor"],
    }

    # Add rule to SCA object instance
    sca.var_rules = var_rules

    passed = sca.passes_variable_rules(fa, var)

    assert passed


def test_passes_variable_rules_specific_failed():
    """Tests the case where a specific variable rule does not apply."""

    # Create SCA object instance
    sca = SpecificationCurveAnalysis()

    fa = {"analysis_type": "simple", "rotation": "oblimin"}
    var = "g_only"

    # Only applies to bifactor
    var_rules = {
        "g_only": [lambda fa: fa["analysis_type"] == "bifactor"],
    }

    # Add rule to SCA object instance
    sca.var_rules = var_rules

    passed = sca.passes_variable_rules(fa, var)

    assert not passed


def test_get_hash():
    """Tests the function that computes the hash of a factor-analysis specification."""

    fa = {"factor_method": "ml", "rotation": "varimax", "n_factors": "3"}
    h = get_hash(fa)

    assert h == "a0e998ac99d5c92d6f598489303eb377"


def test_run_or_load_regression_load(monkeypatch):
    """Tests the function that gets the regression model.

    In this case, we pretend the file exists, and we load the data.
    """

    # Create input variables
    reg_name = "regression_11"
    reg_spec = {"beta_0": True, "beta_1": True, "beta_2": False}
    reg_vars = type("RegVars", (), {"n_sp": 5})()  # mock object
    df_for = pd.DataFrame()

    # Define what should be "loaded" instead of running model
    mock_loaded = pd.DataFrame({"beta_1": [1, 2, 3]})

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

    df_reg = run_or_load_regression(reg_name, reg_spec, reg_vars, df_for)

    # Test whether we return the loaded data
    assert df_reg.equals(pd.DataFrame(mock_loaded))


def test_run_or_load_regression_run(monkeypatch):
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
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.safe_save_dataframe",
        mock_save_dataframe_run_model,
    )

    # Define mock class to replace RegressionFor
    class MockRegression:
        def __init__(self, reg_vars):
            self.reg_vars = reg_vars

        def parallel_estimation(self, df_for, prior_columns):
            return mock_result

    # Import whole SCA file, not just class for the following patch
    from FOR_1_Paper.sca import SpecificationCurveAnalysis

    # Patch RegressionFor
    monkeypatch.setattr(
        SpecificationCurveAnalysis, "RegressionFor", MockRegression, raising=True
    )
    df_reg = run_or_load_regression(reg_name, reg_spec, reg_vars, df_for)

    # Test whether we return the loaded data
    assert saved["name"] == "regression_11_5sp_beab49c1afc5dda7242065f6e0547ef5"
    assert saved["df"].equals(mock_result)
    assert df_reg.equals(mock_result)


def test_run_or_load_rbm_load(monkeypatch):
    """Tests the function that gets the RBM.

    In this case, we pretend the file exists, and we load the data.
    """

    # Create input variables
    rbm_name = "rbm_11"
    rbm_spec = {"h": True, "s": True, "u": False}
    est_vars = type("EstVars", (), {"n_sp": 5})()  # mock object
    df_for = pd.DataFrame()

    agent_vars = AgentVars()

    # Define what should be "loaded" instead of running model
    mock_loaded = pd.DataFrame({"h": [1, 2, 3]})

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

    df_rbm = run_or_load_rbm(rbm_name, rbm_spec, est_vars, agent_vars, df_for)

    # Test whether we return the loaded data
    assert df_rbm.equals(pd.DataFrame(mock_loaded))


def test_run_or_load_rbm_run(monkeypatch):
    """Tests the function that gets the RBM.

    In this case, the file does not exist, so we pretend to run the model.
    """

    # Mock function for data saving
    saved = {}

    def mock_save_dataframe_run_rbm(df):
        """Mock function for data saving."""

        saved["name"] = getattr(df, "name", None)
        saved["df"] = df.copy()

    # Create input variables
    rbm_name = "rbm_11"
    rbm_spec = {"h": True}
    est_vars = ForEstVars()
    df_for = pd.DataFrame()
    agent_vars = AgentVars()

    # Define what should be the result of the regression that we "run"
    mock_result = pd.DataFrame({"h": [2, 3, 4]})

    # Patch save_safe_dataframe function so we can check the saved data
    monkeypatch.setattr(
        "FOR_1_Paper.sca.SpecificationCurveAnalysis.safe_save_dataframe",
        mock_save_dataframe_run_rbm,
    )

    # Define mock class to replace ForEstimation
    class MockEstimation:
        def __init__(self, est_vars):
            self.est_vars = est_vars

        def parallel_estimation(self, df_for, prior_columns):
            return mock_result

    # Import whole SCA file, not just class for the following patch
    from FOR_1_Paper.sca import SpecificationCurveAnalysis

    # Patch RegressionFor
    monkeypatch.setattr(
        SpecificationCurveAnalysis, "ForEstimation", MockEstimation, raising=True
    )
    df_rbm = run_or_load_rbm(rbm_name, rbm_spec, est_vars, agent_vars, df_for)

    # Test whether we return the loaded data
    assert saved["name"] == "rbm_11_10sp_a650037fb49069460a4909d0bf345456"
    assert saved["df"].equals(mock_result)
    assert df_rbm.equals(mock_result)


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


def test_check_fa_spec():
    """Tests the factor-analysis specification checker, which returns "False" for bifactor with varimax rotation."""

    fa = {"analysis_type": "simple", "rotation": "oblimin"}
    allowed = check_fa_spec(fa)
    assert allowed

    fa = {"analysis_type": "bifactor", "rotation": "varimax"}
    allowed = check_fa_spec(fa)
    assert not allowed


# ----------------
# Helper functions
# ----------------


def _fname(h: str) -> str:
    """Helper function that creates a file name from a hash."""

    return f"sca_fa_{h}.pkl"


def _touch(folder: Path, name: str) -> None:
    """Helper function that creates an empty file in the given folder with the given name."""

    (folder / name).write_text("{}", encoding="utf-8")


def make_analysis_df(n=65, which_var="beta_1"):
    """Creates a mock dataframe for unit testing."""

    # IDs
    ids = np.arange(1, n + 1)

    # Betas: this will have a perfect correlation with questionnaire data:
    beta = np.linspace(0, 1, n)
    bic = np.linspace(-10, 0, n)
    df = pd.DataFrame({"ID": ids, which_var: beta, "BIC": bic})
    return df


def make_df_sca_quest_with_id(n=65, var_name="F1"):
    """Creates a mock dataframe for the questionnaire data with ID for unit testing."""

    # IDs
    ids = np.arange(1, n + 1)

    # Factor score: linearly related to beta
    factor = np.linspace(0, 1, n) * 2 + 1
    df = pd.DataFrame({"ID": ids, var_name: factor})
    return df


def make_df_sca_quest_with_subj_num(n=65, var_name="F1"):
    """Creates a mock dataframe for the questionnaire data with subj_num for unit testing."""

    # Create dataframe
    df = make_df_sca_quest_with_id(n, var_name)

    # Rename ID to subj_num
    df = df.rename(columns={"ID": "subj_num"})
    return df


def mock_get_hash(obj):
    """Mock function to replace get_hash."""

    # Depending on input return different hashes
    if isinstance(obj, dict) and "fa" in obj:
        return "FAHASH"
    elif isinstance(obj, dict) and "h" in obj["analysis"]:
        return "RBMHASH"
    return "REGHASH"


# Initialize dict for mock save
saved = {}


def mock_save_dataframe(df, data_dir, print_action=False):
    """Mock function to replace safe_save_dataframe."""

    df_name = getattr(df, "name", "unknown")
    saved[df_name] = {
        "name": getattr(df, "name", None),
        "data_dir": data_dir,
        "columns": list(df.columns),
        "row": df.iloc[0].to_dict(),
    }
