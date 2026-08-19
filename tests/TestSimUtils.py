"""Unit and integration tests for sim_utils.py"""

import random

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import scipy.stats as stats

from FOR_1_Paper.simulations.sim_utils import (generate_questionnaire_data,
                                               get_sim_params, normal_ogive,
                                               sample_hazard_rate,
                                               simulate_and_run_regression)


def test_normal_ogive():
    """Tests normal-ogive (probit) model."""

    # Latent traits
    theta = np.array([-2, 0, 1.5])

    # Item parameters
    a = 1.2  # discrimination (slope)
    taus = np.array([-1.5, -0.5, 0.5, 1.5])  # ordered thresholds for 5 categories

    # Extended thresholds with -inf and +inf
    taus_ext = np.concatenate(([-np.inf], taus, [np.inf]))

    probs = normal_ogive(taus_ext, a, theta)

    # For each theta (latent trait), probabilities of the different response categories have to sum up to 1
    expected_sum = np.array([1.0, 1.0, 1.0])

    # Expected result based on the above parameters
    expected_probs = np.array(
        [
            [0.7257468822499265, 0.03593031911292577, 0.00015910859015755285],
            [0.23832279863714778, 0.23832279863714778, 0.008038427334438603],
            [0.03458042108129571, 0.45149376449985285, 0.10687213429711206],
            [0.001336552282614187, 0.23832279863714778, 0.3849303297782918],
            [1.334574901590631e-05, 0.03593031911292581, 0.5],
        ]
    )

    comp = probs == expected_probs

    assert np.sum(probs, axis=0) == pytest.approx(expected_sum)
    assert comp.all()


def test_generate_questionnaire_data():
    """Test the function that generates questionnaire data based on randomly sampled taus, a, and thetas."""

    n_subj = 2
    n_items = 2

    # Control random number generator for reproducible results
    seed = 123
    np.random.seed(seed)

    questionnaire_data = generate_questionnaire_data(n_subj, n_items=n_items)

    expected_questionnaire_data = pd.DataFrame(
        {
            "subj_num": [1, 1, 2, 2],
            "ID": ["sim_000", "sim_000", "sim_001", "sim_001"],
            "item": [0, 1, 0, 1],
            "response": [3, 1, 5, 2],
            "questionnaire": [1, 1, 1, 1],
            "theta": [
                -0.2714076508251403,
                -0.2714076508251403,
                0.37284740653107196,
                0.37284740653107196,
            ],
            "eta": [0.249336, 0.249336, -0.159725, -0.159725],
        }
    )

    pdt.assert_frame_equal(
        questionnaire_data,
        expected_questionnaire_data,
        check_exact=False,  # approximate float comparison
        atol=1e-5,
    )


def test_sample_hazard_rate_default():
    """Tests sample_hazard_rate function.

    Use default parameters.
    """

    # Control random number generator for reproducible results
    seed = 123
    np.random.seed(seed)

    # Latent traits
    theta = pd.Series([-2, 0, 1.5])

    # Parameters for hazard-rate sampling
    intercept = 0
    slope = 2
    sigma = 1

    hazard_rate = sample_hazard_rate(theta, intercept, slope, sigma)
    expected_hazard_rate = [
        0.006146966687587012,
        0.7305363417397499,
        0.9638402344811637,
    ]

    assert hazard_rate == pytest.approx(expected_hazard_rate)


def test_sample_hazard_rate_no_noise():
    """Tests sample_hazard_rate function.

    No noise in the parameters.
    """

    # Control random number generator for reproducible results
    seed = 123
    np.random.seed(seed)

    # Latent traits
    theta = pd.Series([-2, 0, 1.5])

    # Parameters for hazard-rate sampling
    intercept = 0
    slope = 2
    sigma = 0

    hazard_rate = sample_hazard_rate(theta, intercept, slope, sigma)
    expected_hazard_rate = [0.01798620996209156, 0.5, 0.9525741268224334]

    assert hazard_rate == pytest.approx(expected_hazard_rate)


def test_sample_hazard_rate_no_noise_correlation():
    """Tests sample_hazard_rate function.

    Test if correlation is approximately = 1 if noise = 0.
    """

    # Control random number generator for reproducible results
    seed = 123
    np.random.seed(seed)

    theta = np.random.uniform(-2, 2, 100)

    # Parameters for hazard-rate sampling
    intercept = 0
    slope = 1
    sigma = 0

    hazard_rate = sample_hazard_rate(theta, intercept, slope, sigma)

    # Compute correlation
    r, _ = stats.pearsonr(theta, hazard_rate)

    assert r >= 0.99


def test_get_sim_params():
    """Tests the function that generates the simulation parameters."""

    n_subj = 1000
    hazard_rate = np.linspace(0, 1, 1000)

    gen_model = get_sim_params(n_subj, hazard_rate)

    assert np.min(gen_model["omikron_0"]) >= 1
    assert np.min(gen_model["omikron_1"]) >= 0
    assert np.max(gen_model["omikron_1"]) < 1
    assert np.min(gen_model["s"]) >= 0
    assert np.max(gen_model["s"]) < 1
    assert np.all(gen_model["u"] == 0)
    assert np.min(gen_model["sigma_H"]) >= 0.01
    assert np.max(gen_model["sigma_H"]) < 0.5
    np.testing.assert_array_almost_equal(gen_model["h"].values, hazard_rate, decimal=5)


def test_simulate_and_run_regression():
    """Simple integration test for the simulate_and_run_regression function."""

    # Control random number generator for reproducible results
    seed = 123
    np.random.seed(seed)
    random.seed(seed)

    n_subj = 10
    n_trials = 400
    hazard_rate = np.linspace(0.1, 0.9, n_subj)
    sim = "task_agent"

    kappa = np.tile(np.repeat([16, 8], n_trials / 2), n_subj)
    sigma = np.sqrt(1 / kappa)
    angular_shield_size = 2 * sigma
    subj_num = np.repeat(np.arange(1, n_subj + 1), n_trials)
    new_block = np.tile(np.concatenate([np.array([1]), np.zeros(99)]), 4 * n_subj)
    v_t = np.tile(np.repeat(0.0, n_trials), n_subj)
    group = np.tile(np.repeat(1, n_trials), n_subj)
    df_exp = pd.DataFrame(
        {
            "subj_num": subj_num,
            "sigma": sigma,
            "kappa_t": kappa,
            "new_block": new_block,
            "v_t": v_t,
            "angular_shield_size": angular_shield_size,
            "group": group,
        }
    )
    df_reg, sim_est_errs, df_sim_sca_data, df_rbm = simulate_and_run_regression(
        n_subj, hazard_rate, df_exp, sim=sim
    )

    # df_reg.to_pickle("for_data/testing/integration_test_df_reg.pkl")
    # sim_est_errs.to_pickle("for_data/testing/integration_test_sim_est_errs.pkl")
    # df_sim_sca_data.to_pickle("for_data/testing/integration_test_df_sim_sca_data.pkl")
    # df_rbm.to_pickle("for_data/testing/integration_test_df_rbm.pkl")

    expected_df_reg = pd.read_pickle("for_data/testing/integration_test_df_reg.pkl")
    expected_sim_est_errs = pd.read_pickle(
        "for_data/testing/integration_test_sim_est_errs.pkl"
    )
    expected_df_sim_sca_data = pd.read_pickle(
        "for_data/testing/integration_test_df_sim_sca_data.pkl"
    )
    expected_df_rbm = pd.read_pickle("for_data/testing/integration_test_df_rbm.pkl")

    pd.testing.assert_frame_equal(df_reg, expected_df_reg)
    pd.testing.assert_frame_equal(sim_est_errs, expected_sim_est_errs)
    pd.testing.assert_frame_equal(df_sim_sca_data, expected_df_sim_sca_data)
    pd.testing.assert_frame_equal(df_rbm, expected_df_rbm)
