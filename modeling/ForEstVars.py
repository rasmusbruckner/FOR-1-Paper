"""EstVars: Initialization of the estimation-object instance."""

import numpy as np


class ForEstVars:
    """This function defines the instance variables unique to each instance."""

    def __init__(self):
        # This function determines the default estimation variables

        # Parameter names for data frame
        self.omikron_0 = "omikron_0"  # motor noise
        self.omikron_1 = "omikron_1"  # learning-rate noise
        self.lambda_0 = "lambda_0"  # perseveration intercept
        self.lambda_1 = "lambda_1"  # perseveration slope
        self.h = "h"  # hazard rate
        self.s = "s"  # surprise sensitivity
        self.u = "u"  # uncertainty underestimation
        self.sigma_H = "sigma_H"  # catch-trial

        # Select staring points (used if rand_sp = False)
        self.omikron_0_x0 = 5.0
        self.omikron_1_x0 = 0.0
        self.lambda_0_x0 = 0.1
        self.lambda_1_x0 = -0.2
        self.h_x0 = 0.1
        self.s_x0 = 0.999
        self.u_x0 = 0.0
        self.sigma_H_x0 = 0.01

        # Select range of random starting point values (used if rand_sp = True)
        self.omikron_0_x0_range = (1, 10)
        self.omikron_1_x0_range = (0.001, 1)
        self.lambda_0_x0_range = (-10, 30)
        self.lambda_1_x0_range = (-0.5, 0.0)
        self.h_x0_range = (0.001, 0.99)
        self.s_x0_range = (0.001, 0.99)
        self.u_x0_range = (1, 10)
        self.sigma_H_x0_range = (0.001, 0.5)

        # Select boundaries for estimation
        self.omikron_0_bnds = (0.1, 20)
        self.omikron_1_bnds = (0.001, 1)
        self.lambda_0_bnds = (-10, 30)
        self.lambda_1_bnds = (-0.5, 0.0)
        self.h_bnds = (0.001, 0.99)
        self.s_bnds = (0.001, 1)
        self.u_bnds = (-2, 15)
        self.sigma_H_bnds = (0.001, 0.5)

        # Free parameter indexes
        self.which_vars = {
            self.omikron_0: True,
            self.omikron_1: True,
            self.lambda_0: False,
            self.lambda_1: False,
            self.h: True,
            self.s: True,
            self.u: True,
            self.sigma_H: True,
        }

        # Fixed values for fixed parameters
        self.fixed_mod_coeffs = {
            self.omikron_0: 10.0,
            self.omikron_1: 0.0,
            self.lambda_0: -10,
            self.lambda_1: -0.5,
            self.h: 0.1,
            self.s: 1.0,
            self.u: 0.0,
            self.sigma_H: 0.0001,
        }

        # Other attributes
        self.n_subj = np.nan  # number of participants
        self.n_ker = 4  # number of kernels for estimation
        self.rand_sp = True  # use of random starting points during estimation
        self.n_sp = 10  # number of starting points
        self.use_prior = True  # prior of uncertainty-underestimation parameter
