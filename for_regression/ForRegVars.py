"""This class specifies the regression variables for the FOR circular regression analysis."""

import numpy as np


class RegVars:
    """This class specifies the RegVars object for a circular regression analysis.

    futuretodo: Consider using a data class here.
    """

    def __init__(self):
        """This function defines the instance variables unique to each instance."""

        # Parameter names for data frame
        self.beta_0 = "beta_0"  # intercept
        self.beta_1 = "beta_1"  # coefficient for prediction error (delta)
        self.beta_2 = "beta_2"  # coefficient for prediction error * change-point probability (omega)
        self.beta_3 = (
            "beta_3"  # coefficient for prediction error * relative uncertainty (tau)
        )
        self.beta_4 = "beta_4"  # coefficient for prediction error * alpha (adaptive LR)
        self.beta_5 = "beta_5"  # coefficient for prediction error * hit (r_t)
        self.beta_6 = "beta_6"  # coefficient for prediction error * noise condition
        self.beta_7 = "beta_7"  # coefficient for prediction error * catch-trial dummy
        self.beta_8 = "beta_8"  # coefficient for estimation error * catch-trial dummy
        self.omikron_0 = "omikron_0"  # noise intercept
        self.omikron_1 = "omikron_1"  # noise slope
        self.lambda_0 = "lambda_0"  # perseveration intercept
        self.lambda_1 = "lambda_1"  # perseveration slope

        # Variable names of update regressors (independent of noise terms)
        self.which_update_regressors = [
            "int",
            "delta_t",
            "omega_t",
            "tau_t",
            "alpha_t",
            "r_t",
            "sigma_t",
            "vis_delta",
            "vis_est",
        ]

        # Select staring points (used if rand_sp = False)
        self.beta_0_x0 = 0
        self.beta_1_x0 = 0
        self.beta_2_x0 = 0
        self.beta_3_x0 = 0
        self.beta_4_x0 = 0
        self.beta_5_x0 = 0
        self.beta_6_x0 = 0
        self.beta_7_x0 = 0
        self.beta_8_x0 = 0
        self.omikron_0_x0 = 5
        self.omikron_1_x0 = 0
        self.lambda_0_x0 = 0.1
        self.lambda_1_x0 = 0.1

        # Select range of random starting point values
        self.beta_0_x0_range = (-0.5, 0.5)
        self.beta_1_x0_range = (0, 1)
        self.beta_2_x0_range = (0, 1)
        self.beta_3_x0_range = (0, 1)
        self.beta_4_x0_range = (-1, 1)
        self.beta_5_x0_range = (-1, 1)
        self.beta_6_x0_range = (-1, 1)
        self.beta_7_x0_range = (-1, 1)
        self.beta_8_x0_range = (-1, 1)
        self.omikron_0_x0_range = (1, 20)
        self.omikron_1_x0_range = (0, 1)
        self.lambda_0_x0_range = (0, 1)
        self.lambda_1_x0_range = (0, 1)

        # Select boundaries for estimation
        self.beta_0_bnds = (-2, 2)
        self.beta_1_bnds = (-2, 2)
        self.beta_2_bnds = (-2, 2)
        self.beta_3_bnds = (-2, 2)
        self.beta_4_bnds = (-2, 2)
        self.beta_5_bnds = (-2, 2)
        self.beta_6_bnds = (-2, 2)
        self.beta_7_bnds = (-2, 2)
        self.beta_8_bnds = (-2, 2)
        self.omikron_0_bnds = (0.1, 20)
        self.omikron_1_bnds = (0.001, 1)
        self.lambda_0_bnds = (0, 1)
        self.lambda_1_bnds = (0, 1)

        self.bnds = [
            self.beta_0_bnds,
            self.beta_1_bnds,
            self.beta_2_bnds,
            self.beta_3_bnds,
            self.beta_4_bnds,
            self.beta_5_bnds,
            self.beta_6_bnds,
            self.beta_7_bnds,
            self.beta_8_bnds,
            self.omikron_0_bnds,
            self.omikron_1_bnds,
            self.lambda_0_bnds,
            self.lambda_1_bnds,
        ]

        # Free parameters
        self.which_vars = {
            self.beta_0: True,
            self.beta_1: True,
            self.beta_2: True,
            self.beta_3: True,
            self.beta_4: False,
            self.beta_5: True,
            self.beta_6: True,
            self.beta_7: True,
            self.beta_8: True,
            self.omikron_0: True,
            self.omikron_1: True,
            self.lambda_0: False,
            self.lambda_1: True,
        }

        # Fixed parameter values
        self.fixed_coeffs_reg = {
            self.beta_0: 0.0,
            self.beta_1: 0.0,
            self.beta_2: 0.0,
            self.beta_3: 0.0,
            self.beta_4: 0.0,
            self.beta_5: 0.0,
            self.beta_6: 0.0,
            self.beta_7: 0.0,
            self.beta_8: 0.0,
            self.omikron_0: 10.0,
            self.omikron_1: 0.0,
            self.lambda_0: 0.0,
            self.lambda_1: 0.0,
        }

        # When prior is used: pior mean
        self.beta_0_prior_mean = 0
        self.beta_1_prior_mean = 0
        self.beta_2_prior_mean = 0
        self.beta_3_prior_mean = 0
        self.beta_4_prior_mean = 0
        self.beta_5_prior_mean = 0
        self.beta_6_prior_mean = 0
        self.beta_7_prior_mean = 0
        self.beta_8_prior_mean = 0
        self.omikron_0_prior_mean = 10
        self.omikron_1_prior_mean = 0.1
        self.lambda_0_prior_mean = 0
        self.lambda_1_prior_mean = 0

        # All prior means
        self.prior_mean = [
            self.beta_0_prior_mean,
            self.beta_1_prior_mean,
            self.beta_2_prior_mean,
            self.beta_3_prior_mean,
            self.beta_4_prior_mean,
            self.beta_5_prior_mean,
            self.beta_6_prior_mean,
            self.beta_7_prior_mean,
            self.beta_8_prior_mean,
            self.omikron_0_prior_mean,
            self.omikron_1_prior_mean,
            self.lambda_0_prior_mean,
            self.lambda_1_prior_mean,
        ]

        # Whenprior is used: pior width
        # Note these can be tuned for future versions
        self.beta_0_prior_width = 5
        self.beta_1_prior_width = 5
        self.beta_2_prior_width = 5
        self.beta_3_prior_width = 5
        self.beta_4_prior_width = 5
        self.beta_5_prior_width = 5
        self.beta_6_prior_width = 5
        self.beta_7_prior_width = 5
        self.beta_8_prior_width = 5
        self.omikron_0_prior_width = 20
        self.omikron_1_prior_width = 5
        self.lambda_0_prior_width = 5
        self.lambda_1_prior_width = 5

        # All prior widths
        self.prior_width = [
            self.beta_0_prior_width,
            self.beta_1_prior_width,
            self.beta_2_prior_width,
            self.beta_3_prior_width,
            self.beta_4_prior_width,
            self.beta_5_prior_width,
            self.beta_6_prior_width,
            self.beta_7_prior_width,
            self.beta_8_prior_width,
            self.omikron_0_prior_width,
            self.omikron_1_prior_width,
            self.lambda_0_prior_width,
            self.lambda_1_prior_width,
        ]

        # Other attributes
        self.n_subj = np.nan  # number of subjects
        self.n_ker = 4  # number of kernels for estimation
        self.seed = 123  # seed for random number generator
        self.show_ind_prog = True  # Update progress bar for each subject (True, False)
        self.rand_sp = True  # 0 = fixed starting points, 1 = random starting points
        self.n_sp = 5  # number of starting points (if rand_sp = 1)
