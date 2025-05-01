if __name__ == "__main__":

    import os

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from all_in import latex_plt, cm2inch
    from scipy.stats import zscore
    import scipy.stats as stats

    from FOR_1_Paper.rt_analyses.RtRegression import RtRegression, get_df_rt_reg
    from FOR_1_Paper.rt_analyses.RtRegVars import RegVars

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

    df_reg = get_df_rt_reg(df_for)
    df_reg = get_df_rt_reg(df_for, ms=True, cutoff=True, cutoff_low=250, cutoff_high=10000)



    # Extract block indexes
    new_block = df_for["new_block"].values  # block change indicator

    # Recode last entry in variables of interest of each block to nan
    # to avoid that data of different participants are mixed
    to_nan = np.zeros(len(new_block))
    to_nan[:-1] = new_block[1:]
    to_nan[-1] = 1  # manually added, because no new block after last trial

    # df_reg['e_t_rad_abs_next'] = np.nan
    # df_reg.loc[df_reg.index[:-1], 'e_t_rad_abs_next'] = df_reg['e_t_rad_abs'].values[1:]
    # df_reg.loc[to_nan == 1, "e_t_rad_abs_next"] = np.nan

    # # wie daten ordnen?
    # for i in range(n_subj):
    #
    #     df_sub = df_for[df_for['subj_num'] == i + 1]
    #




    df_reg = df_reg.dropna(subset=['delta_t_rad_abs']).reset_index(drop=True)

    #x = x.apply(zscore)

    #y = pd.DataFrame(df_sub['RT'][:].reset_index(drop=True))
    #y = y * 100

    # 1. No log-transformation
    #    No z-scoring
    # ------------------------

    do_zscore = False
    log_rt = False #True #Fals

    # todo:
    #   - verteilung
    #   - PPC
    # todo: plot descriptives/ model agnostic
    # distribution plots for CP/no-CP and other interesting categories (high/low PE, or catch etc.)

    reg_vars = RegVars()
    reg_vars.n_subj = n_subj


    rt_regression = RtRegression(reg_vars)


    r_sq = list()
    # Change point
    sm_formula = 'RT ~ c_t'
    rt_regression.first_level(df_reg.copy(), sm_formula, do_zscore, zscore, log_rt)
    rt_regression.results_second_level()
    behav_labels = ['c_t']
    #rt_regression.plot_second_level(behav_labels)
    df_second_level = rt_regression.df_second_level



    df_rsq = pd.DataFrame(columns = ['r_sq'])
    r_sq.append(rt_regression.df_second_level.loc['r_sq', 'mean'].copy())

    # todo: zum plotten diese funcs nehmen, die ich schon habe.. optional p-wert drüber...
    # und noch weiteren plot von versch r-squares...

    # Model 1:
    sm_formula = 'RT ~ delta_t_rad_abs'
    rt_regression.first_level(df_reg.copy(), sm_formula, do_zscore, zscore, log_rt)
    rt_regression.results_second_level()
    behav_labels = ['delta_t_rad_abs']
    #rt_regression.plot_second_level(behav_labels)
    df_second_level = rt_regression.df_second_level
    r_sq.append(rt_regression.df_second_level.loc['r_sq', 'mean'].copy())


    # Model 2:
    sm_formula = 'RT ~ a_t_rad_abs'
    rt_regression.first_level(df_reg.copy(), sm_formula, do_zscore, zscore, log_rt)
    rt_regression.results_second_level()
    behav_labels = ['a_t_rad_abs']
    #rt_regression.plot_second_level(behav_labels)
    #a = rt_regression.df_second_level
    df_second_level = rt_regression.df_second_level
    r_sq.append(rt_regression.df_second_level.loc['r_sq', 'mean'].copy())

    # Model 3:
    sm_formula = 'RT ~ omega_t'
    rt_regression.first_level(df_reg.copy(), sm_formula, do_zscore, zscore, log_rt)
    rt_regression.results_second_level()
    behav_labels = ['omega_t']
    #rt_regression.plot_second_level(behav_labels)
    df_second_level = rt_regression.df_second_level
    r_sq.append(rt_regression.df_second_level.loc['r_sq', 'mean'].copy())

    # Model 4:
    sm_formula = 'RT ~ tau_t'
    rt_regression.first_level(df_reg.copy(), sm_formula, do_zscore, zscore, log_rt)
    rt_regression.results_second_level()
    behav_labels = ['tau_t']
   # rt_regression.plot_second_level(behav_labels)
    df_second_level = rt_regression.df_second_level
    r_sq.append(rt_regression.df_second_level.loc['r_sq', 'mean'].copy())

    # Model 5:
    sm_formula = 'RT ~ tau_t + omega_t'
    rt_regression.first_level(df_reg.copy(), sm_formula, do_zscore, zscore, log_rt)
    rt_regression.results_second_level()
    behav_labels = ['tau_t', 'omega_t']
    #rt_regression.plot_second_level(behav_labels)
    df_second_level = rt_regression.df_second_level
    r_sq.append(rt_regression.df_second_level.loc['r_sq', 'mean'].copy())

    # Model 6:
    sm_formula = 'RT ~ delta_t_rad_abs + tau_t + omega_t'
    rt_regression.first_level(df_reg.copy(), sm_formula, do_zscore, zscore, log_rt)
    rt_regression.results_second_level()
    behav_labels = ['delta_t_rad_abs', 'tau_t', 'omega_t']
    rt_regression.plot_second_level(behav_labels, grid_size=(2, 2))
    df_second_level = rt_regression.df_second_level
    r_sq.append(rt_regression.df_second_level.loc['r_sq', 'mean'].copy())
    savename = "figures/model_6.png"
    plt.savefig(savename, transparent=False, dpi=400)
    # Model 7:
    sm_formula = 'RT ~ delta_t_rad_abs + tau_t + omega_t + delta_t_rad_abs * tau_t + delta_t_rad_abs * omega_t'
    rt_regression.first_level(df_reg.copy(), sm_formula, do_zscore, zscore, log_rt)
    rt_regression.results_second_level()
    behav_labels = ['delta_t_rad_abs', 'tau_t', 'omega_t', 'delta_t_rad_abs:tau_t', 'delta_t_rad_abs:omega_t']
    rt_regression.plot_second_level(behav_labels, grid_size=(2, 3))
    #print(rt_regression.model.summary())
    df_second_level = rt_regression.df_second_level
    r_sq.append(rt_regression.df_second_level.loc['r_sq', 'mean'].copy())
    savename = "figures/model_7.png"
    plt.savefig(savename, transparent=False, dpi=400)

    # Model 8:
    sm_formula = 'RT ~ delta_t_rad_abs + tau_t + omega_t + a_t_rad_abs'
    rt_regression.first_level(df_reg.copy(), sm_formula, do_zscore, zscore, log_rt)
    rt_regression.results_second_level()
    behav_labels = ['delta_t_rad_abs', 'tau_t', 'omega_t', 'a_t_rad_abs']
    #rt_regression.plot_second_level(behav_labels, grid_size=(2, 3))
    ##print(rt_regression.model.summary())
    df_second_level = rt_regression.df_second_level
    r_sq.append(rt_regression.df_second_level.loc['r_sq', 'mean'].copy())

    df_rsq = pd.DataFrame({'r_sq': r_sq})

    # todo: noise condition als prediktor..

    fig_width = 15
    fig_height = 10
    # Create figure
    f = plt.figure(figsize=cm2inch(fig_width, fig_height))
    #plt.figure()
    plt.bar(np.arange(len(r_sq)), r_sq)
    plt.xlabel("Model")
    plt.ylabel("R Square")
    sns.despine()
    savename = "figures/r_square.png"
    plt.savefig(savename, transparent=False, dpi=400)
    plt.ioff()
    plt.show()
    # R-square




