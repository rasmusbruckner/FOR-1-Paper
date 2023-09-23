""" FOR circular regression: This script runs a simple circular regression analysis on the FOR dateset

    1. Load data
    2. Run regression analysis
    3. Plot the results

"""

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import os
    from al_plot_utils import cm2inch, label_subplots
    from ForRegVars import RegVars
    from RegressionFor import RegressionFor

    # Turn interactive mode on
    plt.ion()

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # ------------
    # 1. Load data
    # ------------

    # Data of follow-up experiment
    df_for = pd.read_pickle('for_bids_data/data_prepr.pkl')
    n_subj = len(np.unique(df_for['subj_num']))  # number of subjects

    # --------------------------
    # 2. Run regression analysis
    # --------------------------

    # Define regression variables
    # ---------------------------

    reg_vars = RegVars()
    reg_vars.n_subj = n_subj  # number of subjects
    reg_vars.n_ker = 4  # number of kernels for estimation
    reg_vars.n_sp = 10  # number of random starting points
    reg_vars.rand_sp = True  # use random starting points

    # Run mixture model
    # -----------------

    # Free parameters
    reg_vars.which_vars = {reg_vars.beta_0: True,  # intercept
                           reg_vars.beta_1: True,  # delta_t
                           reg_vars.omikron_0: True,  # motor noise
                           reg_vars.omikron_1: True,  # learning-rate noise
                           }

    # Select parameters according to selected variables and create data frame
    prior_columns = [reg_vars.beta_0, reg_vars.beta_1, reg_vars.omikron_0, reg_vars.omikron_1]

    # Initialize regression object
    for_regression = RegressionFor(reg_vars)  # regression object instance

    # Translate degrees to radians, which is necessary for our regression model
    df_for['a_t'] = np.deg2rad(df_for['a_t'])
    df_for['delta_t'] = np.deg2rad(df_for['delta_t'])

    # Drop nans
    df_for = df_for.dropna(subset=['delta_t', 'a_t']).reset_index()

    # Run regression
    # --------------

    results_df = for_regression.parallel_estimation(df_for, prior_columns)

    # -------------------
    # 3. Plot the results
    # -------------------

    # Size of figure
    fig_height = 7
    fig_width = 15

    # Create figure
    f = plt.figure(figsize=cm2inch(fig_width, fig_height))

    # Create plot grid
    gs_0 = gridspec.GridSpec(1, 1)

    # Create subplot grid
    gs_00 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_0[0], wspace=0.75)

    # Plot intercept swarm-boxplot
    # ----------------------------

    ax_0 = plt.Subplot(f, gs_00[0, 0])
    f.add_subplot(ax_0)
    sns.boxplot(y="beta_0", data=results_df,
                notch=False, showfliers=False, linewidth=0.8, width=0.15,
                boxprops=dict(alpha=1), ax=ax_0, showcaps=False)
    sns.swarmplot(y="beta_0", data=results_df, color='gray', alpha=0.7, size=3, ax=ax_0)

    # Plot learning-rate swarm-boxplot
    # --------------------------------

    ax_1 = plt.Subplot(f, gs_00[0, 1])
    f.add_subplot(ax_1)
    sns.boxplot(y="beta_1", data=results_df,
                notch=False, showfliers=False, linewidth=0.8, width=0.15,
                boxprops=dict(alpha=1), ax=ax_1, showcaps=False)
    sns.swarmplot(y="beta_1", data=results_df, color='gray', alpha=0.7, size=3, ax=ax_1)

    # Plot motor-noise swarm-boxplot
    # ------------------------------

    ax_2 = plt.Subplot(f, gs_00[0, 2])
    f.add_subplot(ax_2)
    sns.boxplot(y="omikron_0", data=results_df,
                notch=False, showfliers=False, linewidth=0.8, width=0.15,
                boxprops=dict(alpha=1), ax=ax_2, showcaps=False)
    sns.swarmplot(y="omikron_0", data=results_df, color='gray', alpha=0.7, size=3, ax=ax_2)

    # Plot learning-rate-noise swarm-boxplot
    # ------------------------------

    ax_2 = plt.Subplot(f, gs_00[0, 3])
    f.add_subplot(ax_2)
    sns.boxplot(y="omikron_1", data=results_df,
                notch=False, showfliers=False, linewidth=0.8, width=0.15,
                boxprops=dict(alpha=1), ax=ax_2, showcaps=False)
    sns.swarmplot(y="omikron_1", data=results_df, color='gray', alpha=0.7, size=3, ax=ax_2)

    # Delete unnecessary axes and tighten layout
    sns.despine()

    # Add subplot labels and save figure
    # ----------------------------------

    # Add labels
    texts = ['a', 'b', 'c', 'd']
    label_subplots(f, texts, x_offset=0.08, y_offset=0.025)

    # Show plot
    plt.ioff()
    plt.show()
