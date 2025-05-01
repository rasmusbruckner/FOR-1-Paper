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

    from FOR_1_Paper.for_utilities import get_df_rt_reg, compute_noise_ceiling


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
    df_reg = get_df_rt_reg(df_for, ms=False, cutoff=True, cutoff_low=0.250, cutoff_high=10)



    reg_vars = RegVars()
    reg_vars.n_subj = n_subj
    reg_vars.n_bootstraps = 100




    noise_ceilings = df_reg.groupby('subj_num').apply(compute_noise_ceiling, rt_col='log_rt').reset_index(name='split_half_r')

    noise_ceilings['noise_ceiling_r2'] = noise_ceilings['split_half_r'] ** 2
    a = 1

    subj_num = 1
    df_subj = df_reg[df_reg['subj_num'] == subj_num].copy()
    print(f"{subj_num} has {len(df_subj)} trials")

    compute_noise_ceiling(df_subj, rt_col='log_rt')


    shuffled = df_subj['log_rt'].sample(frac=1).values
    half = len(shuffled) // 2
    rt1 = shuffled[:half]
    rt2 = shuffled[half:half+half]

    import matplotlib.pyplot as plt
    plt.scatter(rt1, rt2, alpha=0.6)
    plt.xlabel("Half A")
    plt.ylabel("Half B")
    plt.title("Split-half RTs for subject S01")
    plt.show()

    print("r:", stats.pearsonr(rt1, rt2))
    print("r:", stats.spearmanr(rt1, rt2))
