"""Simple plot of regression results."""

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
    import pandas as pd
    from allinpy import latex_plt, label_subplots
    from rbmpy import parameter_summary

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # ---------
    # Load data
    # ---------

    model_2 = pd.read_pickle("for_data/regression_23_50sp.pkl")

    # ------------
    # Plot results
    # ------------

    behav_labels = [
        "beta_0",
        "beta_1",
        "beta_4",
        "beta_5",
        "beta_6",
        "omikron_0",
        "omikron_1",
    ]

    axis_labels = [
        "Intercept",
        "Fixed learning rate",
        "Adaptive learning rate",
        "Hit",
        "Noise condition",
        "Motor noise",
        "Learning-rate noise",
    ]

    grid_size = (3, 3)
    fig_size = (15, 12)
    f = parameter_summary(model_2, behav_labels, grid_size, axis_labels=axis_labels, fig_size=fig_size)

    # ----------------------------------
    # Add subplot labels and save figure
    # ----------------------------------

    texts = ["a", "b", "c", "d", "e", "f", "g", "h"]  # label letters
    label_subplots(f, texts, 0.07, y_offset=0.01)

    # Save figure
    save_name = (
            "/"
            + home_dir
            + "/rasmus/Dropbox/Apps/Overleaf/FOR-1-Paper/Figures/model_param_figure_regression.pdf"
    )
    plt.savefig(save_name, dpi=400)

    # Show plot
    plt.ioff()
    plt.show()
