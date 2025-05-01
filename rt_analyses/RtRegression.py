import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import zscore
import numpy as np
import scipy.stats as stats
from rbm_analyses import parameter_summary
from time import sleep
from tqdm import tqdm
from all_in import callback




class RtRegression:

    def __init__(self, reg_vars):

        # todo: hier schauen was alles generell ist

        self.n_subj = reg_vars.n_subj
        self.model = np.nan
        self.n_bootstraps = reg_vars.n_bootstraps
        self.show_ind_prog = reg_vars.show_ind_prog

    # todo: hier schauen was ich in self packe
    def first_level(self, df, sm_formula, do_zscore, zscore, log_rt):

        df = df.copy()
        #pd.testing.assert_frame_equal(df_for, original_df_for)

        # todo: einfach in skript definieren..
        if log_rt:
            df['RT'] = np.log(df['RT'])

        # n_bootstraps = 1000
        postpred_predictions = []
        postpred_subj_num = []
        postpred_trial = []

        # wie daten ordnen?
        for i in range(self.n_subj):

            df_sub = df[df['subj_num'] == i + 1]

            # plt.figure()
            # plt.hist(np.log(df_sub['RT']), 20)
            # plt.ioff()
            # plt.show()

            #x = pd.DataFrame(abs(df_sub[var][:]).reset_index(drop=True))

            #if do_zscore:
            #    x = x.apply(zscore)

            #x = sm.add_constant(x)
            #y = pd.DataFrame(df_sub['RT'][:].reset_index(drop=True))
            #y = y * 100

            #  if do_zscore:
            #      y = y.apply(zscore)

            #model = sm.OLS(y, x).fit()
            self.model = smf.ols(formula=sm_formula, data=df_sub).fit()

            preds = self.model.predict(df_sub)

            # Todo: build huge df with all relevant stuff

            # coeff[i] = model.params[var]
            # print(model.summary())

            df_model_sub = pd.DataFrame(self.model.params.copy())
            df_model_sub.name = 'df_model'
            df_model_sub = df_model_sub.transpose()
            df_model_sub['r_sq'] = self.model.rsquared.copy()

            # Save predictions and coefficients
            postpred_predictions.extend(preds)
            postpred_subj_num.extend(np.repeat(i + 1, len(df_sub)))
            postpred_trial.extend(df_sub.index.values)

            if i == 0:
                self.df_model = df_model_sub
            else:
                self.df_model = pd.concat([self.df_model, df_model_sub], ignore_index=True)

        # todo. vielleicht output
        self.df_postpred = pd.DataFrame({'subj_num': postpred_subj_num,
                                     'trial': postpred_trial,
                                     'pred': postpred_predictions})

    def results_second_level(self):

        self.df_second_level = pd.DataFrame(columns=['mean', 'sd', 'sem', 't_value', 'p_value'])
        self.df_second_level['mean'] = self.df_model.mean()
        self.df_second_level['sd'] = self.df_model.std()
        self.df_second_level['sem'] = self.df_model.std() / np.sqrt(self.n_subj)

        for column_name, column_data in self.df_model.items():
            #print(f"{column_name}:\n{column_data}")

            ttest_result = stats.ttest_1samp(column_data, 0)
            #df_second_level[column_name] = ttest_result[1]
            self.df_second_level.loc[column_name, 't_value'] = ttest_result.statistic
            self.df_second_level.loc[column_name, 'p_value'] = ttest_result.pvalue

        a = 1

        #print(self.df_model.mean())
        #print()
        #return df_second_level

    def run_bootstrap(self, df, sm_formula):

        df = df.copy()
        #pd.testing.assert_frame_equal(df_for, original_df_for)

        # # todo: einfach in skript definieren..
        # if log_rt:
        #     df['RT'] = np.log(df['RT'])

        # Inform user about progress
        pbar = None
        if self.show_ind_prog:
            # Inform user
            sleep(0.1)
            print("\nBootstrapping:")
            sleep(0.1)

            # Initialize progress bar
            pbar = tqdm(total=self.n_subj)

        # n_bootstraps = 1000
        bootstrap_predictions = []
        bootstrap_subj_num = []
        bootstrap_boot_num = []
        bootstrap_trial = []

        # wie daten ordnen?
        for i in range(self.n_subj):


            df_sub = df[df['subj_num'] == i + 1]



            for b in range(self.n_bootstraps):
                # dann
                # Resample the data with replacement
                boot_sample = df_sub.sample(n=len(df_sub), replace=True).reset_index(drop=True)

                # Fit model to bootstrap sample
                boot_model = smf.ols(formula=sm_formula, data=boot_sample).fit()

                # Predict on the original data X (important!)
                preds = boot_model.predict(df_sub)

                # Add simulated residual noise
                residual_std = np.sqrt(boot_model.scale)
                noisy_preds = preds + np.random.normal(0, residual_std, size=len(preds))

                # df_model_sub = pd.DataFrame(boot_model.params.copy())
                # df_model_sub.name = 'df_model'
                # df_model_sub = df_model_sub.transpose()
                # df_model_sub['r_sq'] = boot_model.rsquared.copy()
                #
                # if i == 0:
                #     self.df_model = df_model_sub
                # else:
                #     self.df_model = pd.concat([self.df_model, df_model_sub], ignore_index=True)

                # todo: trial num hinzufügen
                # Save predictions and coefficients
                bootstrap_predictions.extend(noisy_preds)
                bootstrap_subj_num.extend(np.repeat(i+1, len(df_sub)))
                bootstrap_boot_num.extend(np.repeat(b, len(df_sub)))
                bootstrap_trial.extend(df_sub.index.values)



                #bootstrap_coefficients.append(boot_model.params.values)
                #bootstrap_sub_num.append

            callback(self.show_ind_prog, pbar)

        # Close progress bar
        if self.show_ind_prog and pbar:
            pbar.close()

        df_bootstrap = pd.DataFrame({'subj_num': bootstrap_subj_num,
                                    'boot_num': bootstrap_boot_num,
                                     'trial': bootstrap_trial,
                                    'pred': bootstrap_predictions})

        return df_bootstrap

    # todo:
    #   nächste funktionen second level stats
    #   ppt
    #   plots

    def plot_second_level(self, behav_labels, grid_size=(3, 4)):

        a = 1
        parameter_summary(self.df_model, behav_labels, grid_size)
        a = 1


def get_df_rt_reg(df_for, ms=True, cutoff=True, cutoff_low=250, cutoff_high=40000):
    df_reg = pd.DataFrame()
    df_reg["c_t"] = df_for["c_t"].copy()
    df_reg["subj_num"] = df_for["subj_num"].copy()
    df_reg["RT"] = df_for["RT"]
    if ms:
        df_reg["RT"] = df_reg["RT"] * 1000

    if cutoff:
        df_reg = df_reg[(df_reg["RT"] >= cutoff_low) & (df_reg["RT"] <= cutoff_high)]

    df_reg["log_rt"] = np.log(df_for["RT"])
    df_reg["a_t_rad_abs"] = abs(df_for["a_t_rad"])
    df_reg["delta_t_rad_abs"] = abs(df_for["delta_t_rad"])
    df_reg["e_t_rad_abs"] = abs(df_for["e_t_rad"])
    df_reg["omega_t"] = df_for["omega_t"]
    df_reg["tau_t"] = df_for["tau_t"]
    df_reg["trial"] = df_for["trial"]
    df_reg["subj_num"] = df_for["subj_num"]

    return df_reg