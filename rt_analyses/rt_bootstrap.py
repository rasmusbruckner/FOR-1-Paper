if __name__ == "__main__":

    import os

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from all_in import latex_plt
    from scipy.stats import zscore
    import scipy.stats as stats
    import statsmodels.formula.api as smf

    from FOR_1_Paper.rt_analyses.RtRegression import RtRegression
    from FOR_1_Paper.rt_analyses.RtRegVars import RegVars

    from FOR_1_Paper.for_utilities import get_df_rt_reg

    import seaborn as sns

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        pass

    # Turn interactive mode on
    plt.ion()

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # ------------
    # 1. Load data
    # ------------

    df_for = pd.read_pickle("for_data/data_prepr_model.pkl")
    n_subj = len(np.unique(df_for["subj_num"]))  # number of subjects


    df_reg = get_df_rt_reg(df_for, ms=False)
    df_reg = get_df_rt_reg(df_for, ms=True)
    df_reg = get_df_rt_reg(df_for, ms=True, cutoff=True, cutoff_low=250, cutoff_high=10000)



    reg_vars = RegVars()
    reg_vars.n_subj = n_subj
    reg_vars.n_bootstraps = 100


    do_zscore = False
    log_rt = False #True #False

    rt_regression = RtRegression(reg_vars)

    rt_type = 'log_rt'
    rt_type = 'RT'
    rt_type = 'log_rt'

    # Update in radians
    #sm_formula = 'RT ~ delta_t_rad_abs + tau_t + omega_t + delta_t_rad_abs * tau_t + delta_t_rad_abs * omega_t'
    sm_formula = rt_type +' ~ delta_t_rad_abs + tau_t + omega_t + delta_t_rad_abs * tau_t + delta_t_rad_abs * omega_t'
    #sm_formula = rt_type +' ~ delta_t_rad_abs'

    rt_regression.first_level(df_reg.copy(), sm_formula, do_zscore, zscore, log_rt)
    rt_regression.results_second_level()
    behav_labels = ['delta_t_rad_abs', 'tau_t', 'omega_t', 'delta_t_rad_abs:tau_t', 'delta_t_rad_abs:omega_t']
    #rt_regression.plot_second_level(behav_labels, grid_size=(2, 3))
    print(rt_regression.model.summary())
    df_second_level = rt_regression.df_second_level

    df_reg = df_reg.dropna(subset=["delta_t_rad_abs"]).reset_index()  # todo: sinnvoll? für consistency?

    from scipy import stats
    df_postpred = rt_regression.df_postpred

    df_postpred = df_postpred.dropna(subset=["pred"]).reset_index()  # todo: sinnvoll? für consistency?

    # Correlate with actual log RT
    r, _ = stats.pearsonr(df_reg[rt_type], df_postpred['pred'])
    print(f"Model correlation: r = {r:.3f}")

    from all_in import cm2inch

    fig_width = 10
    fig_height = 10
    # Create figure
    f = plt.figure(figsize=cm2inch(fig_width, fig_height))
    plt.scatter(df_reg[rt_type], df_postpred['pred'], alpha=0.6, label='Predicted mean')
    plt.plot([df_reg[rt_type].min(), df_reg[rt_type].max()],
             [df_reg[rt_type].min(), df_reg[rt_type].max()],
             color='gray', linestyle='--', label='Ideal')
    plt.title(f"Across Subjects: Correlation: r = {r:.3f}")
    plt.xlabel("Actual log RT")
    plt.ylabel("Predicted log RT")
    sns.despine()
    savename = "figures/postpred.png"
    plt.savefig(savename, transparent=False, dpi=400)
    #plt.ioff()
    #plt.show()



    # For subjects:

    for i in range(n_subj):
        df_postpred = rt_regression.df_postpred

        df_postpred_sub = df_postpred[df_postpred['subj_num'] == i+1].reset_index(drop=True)

        df_postpred_sub = df_postpred_sub.dropna(subset=["pred"]).reset_index(drop=True)  # todo: sinnvoll? für consistency?

        df_sub = df_reg[df_reg['subj_num'] == i+1].reset_index(drop=True)
        df_sub = df_sub.dropna(subset=["delta_t_rad_abs"]).reset_index(drop=True)  # todo: sinnvoll? für consistency?

        fig_width = 10
        fig_height = 10
        # Create figure
        f = plt.figure(figsize=cm2inch(fig_width, fig_height))
        plt.scatter(df_sub[rt_type], df_postpred_sub['pred'], alpha=0.6, label='Predicted mean')
       # plt.plot([df_sub[rt_type].min(), df_sub[rt_type].max()],
        #         [df_sub[rt_type].min(), df_sub[rt_type].max()],
         #        color='gray', linestyle='--', label='Ideal')

        # Fit a line
        x = df_sub[rt_type]
        y = df_postpred_sub['pred']
        slope, intercept = np.polyfit(x, y, 1)
        plt.plot(x, slope * x + intercept, color='red', label='Linear fit')

        plt.title(f"Across Subjects: Correlation: r = {r:.3f}")
        plt.xlabel("Actual log RT")
        plt.ylabel("Predicted log RT")

        # Correlate with actual log RT
        r, _ = stats.pearsonr(df_sub[rt_type], df_postpred_sub['pred'])
        print(f"Model correlation: r = {r:.3f}")
        plt.title(f"Within Subject: Correlation: r = {r:.3f}")


        sns.despine()
        savename = "figures/single_sub/postpred_sub_" + str(i+1) + ".png"
        plt.savefig(savename, transparent=False, dpi=400)
        plt.ioff()
        #plt.show()
        plt.close()
        #
    #
    # df_bootstrap = rt_regression.run_bootstrap(df_reg.copy(), sm_formula)
    #
    # # Drop nans
    # df_bootstrap = df_bootstrap.dropna(subset=["pred"]).reset_index()
    #
    #
    # # Group by subject and trial
    # summary_df = df_bootstrap.groupby(['subj_num', 'trial'])['pred'].agg([
    #     ('pred_mean', 'mean'),
    #     ('pred_lower', lambda x: x.quantile(0.025)),
    #     ('pred_upper', lambda x: x.quantile(0.975))
    # ]).reset_index()
    #
    # actual_rts = df_reg[['subj_num', 'trial', rt_type]].copy()
    #
    # full_df = summary_df.merge(actual_rts, on=['subj_num', 'trial'])
    #
    #
    # plt.figure(figsize=(6, 6))
    # plt.scatter(full_df[rt_type], full_df['pred_mean'], alpha=0.6, label='Predicted mean')
    # plt.plot([full_df[rt_type].min(), full_df[rt_type].max()],
    #          [full_df[rt_type].min(), full_df[rt_type].max()],
    #          color='gray', linestyle='--', label='Ideal')
    #
    # # Optional: add prediction intervals
    # plt.errorbar(full_df[rt_type], full_df['pred_mean'],
    #              yerr=[full_df['pred_mean'] -full_df['pred_lower'], full_df['pred_upper'] - full_df['pred_mean']],
    #              fmt='o', alpha=0.2, color='orange', label='95% PI')
    #
    # plt.xlabel('Actual RT')
    # plt.ylabel('Predicted RT (mean of bootstraps)')
    # plt.title('Actual vs Predicted RT (Full Model)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # a = 1
    #


