"""Supplementary figure regression recovery.

Always make sure the manual axis limits don't hide any data points.
"""

if __name__ == "__main__":

    import os
    import platform

    import matplotlib

    # Simple cross-platform backend selection
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")  # headless
    elif platform.system() == "Darwin":
        matplotlib.use("MacOSX")  # macOS native
    else:
        matplotlib.use("Qt5Agg")  # Linux with display, Windows, others

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from allinpy import cm2inch, label_subplots, latex_plt

    from FOR_1_Paper.for_utilities import plot_param

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    # ------------
    # 1. Load data
    # ------------

    recovered_params = pd.read_pickle("for_data/regression_recovery_50_sp.pkl")
    true_params = pd.read_pickle("for_data/regression_recovery_50_sp_true_params.pkl")

    # -----------------
    # 2. Prepare figure
    # -----------------

    # Create figure
    fig_height = 15
    fig_width = 15
    f = plt.figure(figsize=cm2inch(fig_width, fig_height))

    # Y-label distance
    ylabel_dist = -0.3

    # ------------------------
    # 3. Plot recovery results
    # ------------------------

    # Intercept
    plt.subplot(331)
    plt.axline((0, 0), slope=1)
    plot_param("beta_0", "Intercept", true_params, recovered_params)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.xticks(np.arange(-0.5, 0.75, 0.25))
    plt.yticks(np.arange(-0.5, 0.75, 0.25))
    plt.ylabel("Recovered parameter")
    plt.gca().yaxis.set_label_coords(ylabel_dist, 0.5)

    # Fixed LR
    plt.subplot(332)
    plt.axline((0, 0), slope=1)
    plot_param("beta_1", "Fixed LR", true_params, recovered_params)
    plt.xlim(0.35, 1.1)
    plt.ylim(0.35, 1.1)

    # Adaptive LR
    plt.subplot(333)
    plt.axline((0, 0), slope=1)
    plot_param("beta_4", "Adaptive LR", true_params, recovered_params)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xticks(np.arange(-0.0, 1.1, 0.5))
    plt.yticks(np.arange(-0.0, 1.1, 0.5))

    # Hit
    plt.subplot(334)
    plt.axline((0, 0), slope=1)
    plot_param("beta_5", "Hit", true_params, recovered_params)
    plt.xlim(-0.2, 0.12)
    plt.ylim(-0.2, 0.12)
    plt.ylabel("Recovered parameter")
    plt.gca().yaxis.set_label_coords(ylabel_dist, 0.5)

    # Condition
    plt.subplot(335)
    plt.axline((0, 0), slope=1)
    plot_param("beta_6", "Condition", true_params, recovered_params)
    plt.xlim(-1.1, 0.12)
    plt.ylim(-1.1, 0.12)
    plt.xticks(np.arange(-1.0, 0.1, 0.5))
    plt.yticks(np.arange(-1.0, 0.1, 0.5))

    # Catch trial
    plt.subplot(336)
    plt.axline((0, 0), slope=1)
    plot_param("beta_7", "Catch trial", true_params, recovered_params)
    plt.xlim(-0.3, 1.2)
    plt.ylim(-0.3, 1.2)
    plt.xticks(np.arange(0, 1.3, 0.4))
    plt.yticks(np.arange(0, 1.3, 0.4))
    plt.xlabel("True parameter")

    # Motor noise
    plt.subplot(337)
    plt.axline((0, 0), slope=1)
    plot_param("omikron_0", "Motor noise", true_params, recovered_params)
    plt.xlim(0, 17)
    plt.ylim(0, 17)
    plt.xlabel("True parameter")
    plt.ylabel("Recovered Parameter")
    plt.gca().yaxis.set_label_coords(ylabel_dist, 0.5)

    # LR noise
    plt.subplot(338)
    plt.axline((0, 0), slope=1)
    plot_param("omikron_1", "LR noise", true_params, recovered_params)
    plt.xlim(-0.02, 0.45)
    plt.ylim(-0.02, 0.45)
    plt.xticks(np.arange(0.0, 0.46, 0.15))
    plt.yticks(np.arange(0.0, 0.46, 0.15))
    plt.xlabel("True parameter")
    plt.gca().yaxis.set_label_coords(ylabel_dist, 0.5)

    # -------------------------------------
    # 4. Add subplot labels and save figure
    # -------------------------------------

    # Adjust space and axes
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5
    )
    sns.despine()

    texts = ["a", "b", "c", "d", "e", "f", "g", "h"]  # label letters
    label_subplots(f, texts, x_offset=0.07, y_offset=0.01)

    # Save figure
    # -----------

    # Save figure
    save_name = (
        "/"
        + home_dir
        + "/rasmus/Dropbox/Apps/Overleaf/FOR-1-Paper/Figures/regression_recovery.pdf"
    )

    plt.savefig(save_name, dpi=400)

    plt.ioff()
    plt.show()
