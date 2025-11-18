"""This child class specifies the specific methods of the FOR circular regression analysis."""

import numpy as np
import pandas as pd
from rbmpy import (RegressionParent, compute_persprob, get_sel_coeffs,
                   residual_fun)


class RegressionFor(RegressionParent):
    """This class specifies the instance variables and methods of the FOR regression analysis."""

    def __init__(self, reg_vars: "RegVars"):
        """This function defines the instance variables unique to each instance.

        See ForRegVars for documentation.

        Parameters
        ----------
        reg_vars : RegVars
            Regression-variables-object instance.
        """

        # Parameters from parent class
        super().__init__(reg_vars)

        # Extract parameter names for data frame
        self.beta_0 = reg_vars.beta_0
        self.beta_1 = reg_vars.beta_1
        self.beta_2 = reg_vars.beta_2
        self.beta_3 = reg_vars.beta_3
        self.beta_4 = reg_vars.beta_4
        self.beta_5 = reg_vars.beta_5
        self.beta_6 = reg_vars.beta_6
        self.beta_7 = reg_vars.beta_7
        self.beta_8 = reg_vars.beta_8
        self.omikron_0 = reg_vars.omikron_0
        self.omikron_1 = reg_vars.omikron_1
        self.lambda_0 = reg_vars.lambda_0
        self.lambda_1 = reg_vars.lambda_1

        # Extract staring points
        self.beta_0_x0 = reg_vars.beta_0_x0
        self.beta_1_x0 = reg_vars.beta_1_x0
        self.beta_2_x0 = reg_vars.beta_2_x0
        self.beta_3_x0 = reg_vars.beta_3_x0
        self.beta_4_x0 = reg_vars.beta_4_x0
        self.beta_5_x0 = reg_vars.beta_5_x0
        self.beta_6_x0 = reg_vars.beta_6_x0
        self.beta_7_x0 = reg_vars.beta_7_x0
        self.beta_8_x0 = reg_vars.beta_8_x0
        self.omikron_0_x0 = reg_vars.omikron_0_x0
        self.omikron_1_x0 = reg_vars.omikron_1_x0
        self.lambda_0_x0 = reg_vars.lambda_0_x0
        self.lambda_1_x0 = reg_vars.lambda_1_x0

        # Extract range of random starting points
        self.beta_0_x0_range = reg_vars.beta_0_x0_range
        self.beta_1_x0_range = reg_vars.beta_1_x0_range
        self.beta_2_x0_range = reg_vars.beta_2_x0_range
        self.beta_3_x0_range = reg_vars.beta_3_x0_range
        self.beta_4_x0_range = reg_vars.beta_4_x0_range
        self.beta_5_x0_range = reg_vars.beta_5_x0_range
        self.beta_6_x0_range = reg_vars.beta_6_x0_range
        self.beta_7_x0_range = reg_vars.beta_7_x0_range
        self.beta_8_x0_range = reg_vars.beta_8_x0_range
        self.omikron_0_x0_range = reg_vars.omikron_0_x0_range
        self.omikron_1_x0_range = reg_vars.omikron_1_x0_range
        self.lambda_0_x0_range = reg_vars.lambda_0_x0_range
        self.lambda_1_x0_range = reg_vars.lambda_1_x0_range

        # Extract boundaries for estimation
        self.beta_0_bnds = reg_vars.beta_0_bnds
        self.beta_1_bnds = reg_vars.beta_1_bnds
        self.beta_2_bnds = reg_vars.beta_2_bnds
        self.beta_3_bnds = reg_vars.beta_3_bnds
        self.beta_4_bnds = reg_vars.beta_4_bnds
        self.beta_5_bnds = reg_vars.beta_5_bnds
        self.beta_6_bnds = reg_vars.beta_6_bnds
        self.beta_7_bnds = reg_vars.beta_7_bnds
        self.beta_8_bnds = reg_vars.beta_8_bnds
        self.omikron_0_bnds = reg_vars.omikron_0_bnds
        self.omikron_1_bnds = reg_vars.omikron_1_bnds
        self.lambda_0_bnds = reg_vars.lambda_0_bnds
        self.lambda_1_bnds = reg_vars.lambda_1_bnds

        # Extract free parameters
        self.which_vars = reg_vars.which_vars

        # Extract fixed parameter values
        self.fixed_coeffs_reg = reg_vars.fixed_coeffs_reg

        # Update regressors (coefficients unrelated to error terms)
        self.update_regressors = list()

    @staticmethod
    def get_datamat(df: pd.DataFrame) -> pd.DataFrame:
        """This function creates the explanatory matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame containing subset of data.

        Returns
        -------
        pd.DataFrame
            Regression data frame.
        """

        reg_df = pd.DataFrame()
        reg_df["int"] = np.ones(len(df))
        reg_df["delta_t"] = df["delta_t_rad"].to_numpy()
        reg_df["tau_t"] = (
            df["tau_t"].to_numpy() * (1 - df["omega_t"].to_numpy())
        ) * df["delta_t_rad"].to_numpy()
        # reg_df['tau_t'] = df['tau_t'].to_numpy() * df['delta_t'].to_numpy()
        reg_df["omega_t"] = df["omega_t"].to_numpy() * df["delta_t_rad"].to_numpy()
        reg_df["alpha_t"] = df["alpha_t"].to_numpy() * df["delta_t_rad"].to_numpy()
        reg_df["r_t"] = df["hit_dummy"].to_numpy() * df["delta_t_rad"].to_numpy()
        reg_df["sigma_t"] = df["kappa_dummy"].to_numpy() * df["delta_t_rad"].to_numpy()
        reg_df["vis_delta"] = df["v_t"].to_numpy() * df["delta_t_rad"].to_numpy()
        reg_df["vis_est"] = df["v_t"].to_numpy() * df["e_t_rad"].to_numpy()
        reg_df["a_t"] = df["a_t_rad"].to_numpy()
        reg_df["group"] = df["group"].to_numpy()
        reg_df["subj_num"] = df["subj_num"].to_numpy()
        reg_df["ID"] = df["ID"].to_numpy()

        return reg_df

    def get_starting_point(self) -> list:
        """This function determines the starting points of the estimation process.

        Returns
        -------
        list
            List with starting points.
        """

        # Put all starting points into list
        if self.rand_sp:

            # Draw random starting points
            x0 = [
                np.random.uniform(self.beta_0_x0_range[0], self.beta_0_x0_range[1]),
                np.random.uniform(self.beta_1_x0_range[0], self.beta_1_x0_range[1]),
                np.random.uniform(self.beta_2_x0_range[0], self.beta_2_x0_range[1]),
                np.random.uniform(self.beta_3_x0_range[0], self.beta_3_x0_range[1]),
                np.random.uniform(self.beta_4_x0_range[0], self.beta_4_x0_range[1]),
                np.random.uniform(self.beta_5_x0_range[0], self.beta_5_x0_range[1]),
                np.random.uniform(self.beta_6_x0_range[0], self.beta_6_x0_range[1]),
                np.random.uniform(self.beta_7_x0_range[0], self.beta_7_x0_range[1]),
                np.random.uniform(self.beta_8_x0_range[0], self.beta_8_x0_range[1]),
                np.random.uniform(
                    self.omikron_0_x0_range[0], self.omikron_0_x0_range[1]
                ),
                np.random.uniform(
                    self.omikron_1_x0_range[0], self.omikron_1_x0_range[1]
                ),
                np.random.uniform(self.lambda_0_x0_range[0], self.lambda_0_x0_range[1]),
                np.random.uniform(self.lambda_1_x0_range[0], self.lambda_1_x0_range[1]),
            ]

        else:

            # Use fixed starting points
            x0 = [
                self.beta_0_x0,
                self.beta_1_x0,
                self.beta_2_x0,
                self.beta_3_x0,
                self.beta_4_x0,
                self.beta_5_x0,
                self.beta_6_x0,
                self.beta_7_x0,
                self.beta_8_x0,
                self.omikron_0_x0,
                self.omikron_1_x0,
                self.lambda_0_x0,
                self.lambda_1_x0,
            ]

        return x0

    def sample_data(
        self,
        df_params: pd.DataFrame,
        n_trials: int,
        all_sub_behav_data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """This function samples the data for simulations.

        Parameters
        ----------
        df_params : pd.DataFrame
            Regression parameters for simulation.
        n_trials : int
            Number of trials
        all_sub_behav_data : pd.DataFrame
             Optional subject behavioral data.

        Returns
        -------
        pd.DataFrame
            Sampled regression updates.
        """

        # Number of simulations
        n_sim = len(df_params["subj_num"])

        # Initialize
        df_sim = pd.DataFrame()  # simulated data

        # Cycle over simulations
        for i in range(0, n_sim):

            # Extract regression coefficients
            coeffs = df_params.iloc[i].to_numpy()

            # Regression variables
            if all_sub_behav_data is None:

                # Randomly generate data
                df_data = pd.DataFrame(
                    {
                        "delta_t_rad": np.random.uniform(-np.pi, np.pi, n_trials),
                        "e_t_rad": np.random.uniform(-np.pi, np.pi, n_trials),
                        "v_t": np.random.binomial(1, 0.1, n_trials),
                        "hit_dummy": np.random.binomial(1, 0.5, n_trials),
                        "kappa_dummy": np.concatenate(
                            [np.zeros(n_trials // 2), np.ones(n_trials // 2)]
                        ),
                        "tau_t": np.random.rand(n_trials),
                        "omega_t": np.random.rand(n_trials),
                        "alpha_t": np.random.rand(n_trials),
                        "a_t_rad": np.full(n_trials, np.nan),
                        "subj_num": np.random.randint(1, 100, n_trials),
                        "ID": np.random.randint(1000, 5000, n_trials),
                        "group": np.zeros(n_trials),
                    }
                )

            else:
                # Optionally based on subject data:

                # Logical index for subject number
                indices = all_sub_behav_data["subj_num"] == i + 1

                # Extract subset for fields dynamically
                sub_behav_data = {
                    key: value[indices].to_numpy()
                    for key, value in all_sub_behav_data.items()
                }

                # Extract regression data
                df_data = pd.DataFrame(
                    {
                        "delta_t_rad": sub_behav_data["delta_t_rad"],
                        "e_t_rad": sub_behav_data["e_t_rad"],
                        "v_t": sub_behav_data["v_t"],
                        "hit_dummy": sub_behav_data["hit_dummy"],
                        "kappa_dummy": sub_behav_data["kappa_dummy"],
                        "tau_t": sub_behav_data["tau_t"],
                        "omega_t": sub_behav_data["omega_t"],
                        "alpha_t": sub_behav_data["alpha_t"],
                        "a_t_rad": np.full(n_trials, np.nan),
                        "subj_num": sub_behav_data["subj_num"],
                        "ID": sub_behav_data["ID"],
                        "group": np.zeros(n_trials),
                    }
                )

                df_data = df_data.dropna(subset=["delta_t_rad"]).reset_index(drop=True)

            # Create design matrix
            datamat = self.get_datamat(df_data)

            # Extract parameters
            sel_coeffs = get_sel_coeffs(
                self.which_vars.items(), self.fixed_coeffs_reg, coeffs
            )

            # Create linear regression matrix
            lr_mat = datamat[self.which_update_regressors].to_numpy()

            # Linear regression parameters
            update_regressors = [
                value
                for key, value in sel_coeffs.items()
                if key not in ["omikron_0", "omikron_1", "lambda_0", "lambda_1"]
            ]

            # Predicted updates
            a_t_hat = lr_mat @ np.array(update_regressors)

            # Residuals
            if self.which_vars["omikron_1"]:

                # Compute updating noise
                concentration = residual_fun(
                    np.abs(a_t_hat), sel_coeffs["omikron_0"], sel_coeffs["omikron_1"]
                )

            else:
                # Motor noise only
                concentration = np.repeat(sel_coeffs["omikron_0"], len(a_t_hat))

            if self.which_vars["lambda_0"] and not self.which_vars["lambda_1"]:

                # Compute perseveration probability
                pers = np.random.binomial(
                    size=len(a_t_hat), n=1, p=sel_coeffs["lambda_0"]
                )
                a_t_hat[pers == 1] = 0

            elif (self.which_vars["lambda_0"] and self.which_vars["lambda_1"]) or (
                not self.which_vars["lambda_0"] and self.which_vars["lambda_1"]
            ):

                # Compute perseveration probability
                pers_prob = compute_persprob(
                    sel_coeffs["lambda_0"], sel_coeffs["lambda_1"], abs(a_t_hat)
                )
                pers = np.random.binomial(size=len(a_t_hat), n=1, p=pers_prob)
                a_t_hat[pers == 1] = 0

            else:
                pers = np.zeros(len(a_t_hat))

            # Sample updates from von Mises distribution
            a_t_hat[pers == 0] = np.random.vonmises(
                a_t_hat[pers == 0], concentration[pers == 0]
            )

            # Store update and ID
            df_data["a_t_rad"] = a_t_hat
            df_data["subj_num"] = np.full(len(df_data), i + 1)

            # Combine all data
            df_sim = pd.concat([df_sim, df_data], ignore_index=True)

        return df_sim
