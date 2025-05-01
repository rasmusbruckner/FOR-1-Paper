"""Estimation Class: This class estimates the reduced Bayesian model."""

import random
from itertools import compress
from multiprocessing import Pool
from time import sleep

import numpy as np
import pandas as pd
from all_in import callback
from for_task_agent_int_rbm import task_agent_int
from rbm_analyses import AlAgent
from rbm_analyses.utilities import compute_bic, get_sel_coeffs
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm


class ForEstimation:
    """This class specifies the instance variables and methods of the parameter estimation."""

    def __init__(self, est_vars: "EstVars"):
        """This function defines the instance variables unique to each instance.

        See AlEstVars for documentation.

        Parameters
        ----------
        est_vars : "EstVars"
            Estimation variables object instance.
        """

        # Parameter names for data frame
        self.omikron_0 = est_vars.omikron_0
        self.omikron_1 = est_vars.omikron_1
        self.h = est_vars.h
        self.s = est_vars.s
        self.u = est_vars.u
        self.sigma_H = est_vars.sigma_H

        # Select fixed staring points (used if not rand_sp)
        self.omikron_0_x0 = est_vars.omikron_0_x0
        self.omikron_1_x0 = est_vars.omikron_1_x0
        self.h_x0 = est_vars.h_x0
        self.s_x0 = est_vars.s_x0
        self.u_x0 = est_vars.u_x0
        self.sigma_H_x0 = est_vars.sigma_H_x0

        # Select range of random starting point values (if rand_sp)
        self.omikron_0_x0_range = est_vars.omikron_0_x0_range
        self.omikron_1_x0_range = est_vars.omikron_1_x0_range
        self.h_x0_range = est_vars.h_x0_range
        self.s_x0_range = est_vars.s_x0_range
        self.u_x0_range = est_vars.u_x0_range
        self.sigma_H_x0_range = est_vars.sigma_H_x0_range

        # Select boundaries for estimation
        self.omikron_0_bnds = est_vars.omikron_0_bnds
        self.omikron_1_bnds = est_vars.omikron_1_bnds
        self.h_bnds = est_vars.h_bnds
        self.s_bnds = est_vars.s_bnds
        self.u_bnds = est_vars.u_bnds
        self.sigma_H_bnds = est_vars.sigma_H_bnds

        # Free parameter indexes
        self.which_vars = est_vars.which_vars

        # Fixed parameter values
        self.fixed_mod_coeffs = est_vars.fixed_mod_coeffs

        # Other attributes
        self.n_subj = est_vars.n_subj
        self.n_ker = est_vars.n_ker
        self.rand_sp = est_vars.rand_sp
        self.n_sp = est_vars.n_sp
        self.use_prior = est_vars.use_prior

    def parallel_estimation(
        self, df: pd.DataFrame, agent_vars: "AgentVArs"
    ) -> pd.DataFrame:
        """This function manages the parallelization of the model estimation.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame containing the data.
        agent_vars : "AgentVars"
            Agent variables object instance.

        Returns
        -------
        pd.DataFrame
            Data frame containing regression results.
        """

        # Inform user
        sleep(0.1)
        print("\nModel estimation:")
        sleep(0.1)

        # Initialize progress bar
        pbar = tqdm(total=self.n_subj)

        # Initialize pool object for parallel processing
        pool = Pool(processes=self.n_ker)

        # Parallel parameter estimation
        results = [
            pool.apply_async(
                self.model_estimation,
                args=(df[(df["subj_num"] == i + 1)].copy(), agent_vars),
                callback=lambda _: callback(True, pbar),
            )
            for i in range(0, self.n_subj)
        ]

        output = [p.get() for p in results]
        pool.close()
        pool.join()

        # Select parameters according to selected variables and create data frame
        prior_columns = [
            self.omikron_0,
            self.omikron_1,
            self.h,
            self.s,
            self.u,
            self.sigma_H,
        ]

        # Add estimation results to data frame output
        columns = list(compress(prior_columns, self.which_vars.values()))
        columns.append("llh")
        columns.append("BIC")
        columns.append("subj_num")
        results_df = pd.DataFrame(output, columns=columns)

        # Make sure that we keep the same order of participants
        results_df = results_df.sort_values(by=["subj_num"])

        # Close progress bar
        pbar.close()

        return results_df

    def model_estimation(self, df_subj: pd.DataFrame, agent_vars: "AgentVars") -> list:
        """This function estimates the free parameters of the model.

        Parameters
        ----------
        df_subj : pd.DataFrame
            Data frame with data of current participants.
        agent_vars : "AgentVars"
            Agent variables object instance.

        Returns
        -------
        list
            List containing estimates, llh, bic and age group.
        """

        # Control random number generator for reproducible results
        random.seed(123)

        # Extract age group and subject number for output
        subj_num = list(set(df_subj["subj_num"]))
        subj_num = float(subj_num[0])

        # Extract free parameters
        values = self.which_vars.values()

        # Select starting points and boundaries
        # -------------------------------------

        bnds = [
            self.omikron_0_bnds,
            self.omikron_1_bnds,
            self.h_bnds,
            self.s_bnds,
            self.u_bnds,
            self.sigma_H_bnds,
        ]

        # Select boundaries according to selected free parameters
        bnds = np.array(list(compress(bnds, values)))

        # Initialize with unrealistically high likelihood
        min_llh = 100000  # futuretodo: set to inf?
        min_x = np.nan

        # Cycle over starting points
        for r in range(0, self.n_sp):

            if self.rand_sp:

                # Draw starting points from uniform distribution
                x0 = [
                    random.uniform(
                        self.omikron_0_x0_range[0], self.omikron_0_x0_range[1]
                    ),
                    random.uniform(
                        self.omikron_1_x0_range[0], self.omikron_1_x0_range[1]
                    ),
                    random.uniform(self.h_x0_range[0], self.h_x0_range[1]),
                    random.uniform(self.s_x0_range[0], self.s_x0_range[1]),
                    random.uniform(self.u_x0_range[0], self.u_x0_range[1]),
                    random.uniform(self.sigma_H_x0_range[0], self.sigma_H_x0_range[1]),
                ]
            else:

                # Use fixed starting points
                x0 = [
                    self.omikron_0_x0,
                    self.omikron_1_x0,
                    self.h_x0,
                    self.s_x0,
                    self.u_x0,
                    self.sigma_H_x0,
                ]

            # Select starting points according to free parameters
            x0 = np.array(list(compress(x0, values)))

            # Estimate parameters
            res = minimize(
                self.llh,
                x0,
                args=(df_subj, agent_vars),
                method="L-BFGS-B",
                bounds=bnds,
                options={"disp": False},
            )  # options={'disp': False}

            # Extract minimized log likelihood
            f_llh_max = res.fun

            # Check if negative log-likelihood is lower than the previous one and select the lowest
            if f_llh_max < min_llh:
                min_llh = f_llh_max
                min_x = res.x

        # Compute BIC todo: subtract nan trials in T trials
        bic = compute_bic(min_llh, sum(self.which_vars.values()), len(df_subj))

        # Save data to list of results
        min_x = min_x.tolist()
        results_list = list()
        results_list = results_list + min_x
        results_list.append(float(min_llh))
        results_list.append(float(bic))
        results_list.append(float(subj_num))

        return results_list

    def llh(
        self, coeffs: np.ndarray, df: pd.DataFrame, agent_vars: "AgentVars"
    ) -> float:
        """This function computes the cumulated negative log-likelihood of the data under the model.

        Parameters
        ----------
        coeffs : np.ndarray
            Free parameters.
        df : pd.DataFrame
            Data frame of current subject.
        agent_vars : AgentVars
            Agent variables object.

        Returns
        -------
        float
            Cumulated negative log-likelihood.
        """

        # Get fixed parameters
        fixed_coeffs = self.fixed_mod_coeffs

        # Extract parameters
        sel_coeffs = get_sel_coeffs(self.which_vars.items(), fixed_coeffs, coeffs)

        df = df.reset_index(drop=True)  # adjust index

        # Reduced Bayesian model variables
        agent_vars.h = sel_coeffs["h"]
        agent_vars.s = sel_coeffs["s"]
        agent_vars.u = np.exp(sel_coeffs["u"])
        agent_vars.sigma_H = sel_coeffs["sigma_H"]

        # Ensure we use the circular model
        agent_vars.circular = True

        # Call AlAgent object
        agent = AlAgent(agent_vars)

        # Estimate parameters
        llh_mix, _ = task_agent_int(df, agent, agent_vars, sel_coeffs)

        # Consider prior over uncertainty-underestimation coefficient
        if self.use_prior:
            u_prob = np.log(norm.pdf(sel_coeffs["u"], 0, 5))
        else:
            u_prob = 0

        # Compute cumulated negative log-likelihood
        llh_sum = -1 * (np.sum(llh_mix) + u_prob)

        return llh_sum
