if __name__ == "__main__":

    import os
    import platform

    import matplotlib

    # Simple cross-platform backend selection
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")  # Headless
    elif platform.system() == "Darwin":
        matplotlib.use("MacOSX")  # macOS native
    else:
        matplotlib.use("Qt5Agg")  # Linux with display, Windows, others

    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from allinpy import cm2inch, label_subplots, latex_plt
    from scipy.stats import norm

    from FOR_1_Paper.simulations.sim_utils import normal_ogive

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # Turn on interactive mode
    plt.ion()

    # --------------
    # Prepare figure
    # --------------

    # Figure properties
    fig_height = 9
    fig_width = 15

    # Initialize figure
    f = plt.figure(figsize=cm2inch(fig_width, fig_height))
    gs0 = gridspec.GridSpec(1, 1, bottom=0.4)
    gs_01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0], wspace=0.3)

    # -----------------------
    # Plot normal-ogive model
    # -----------------------

    # Plot response probabilities
    # ---------------------------

    # Latent trait
    theta = np.linspace(-3.5, 3.5, 600)

    # Item parameters
    a = 0.8  # discrimination
    taus = np.array([-1.5, -0.5, 0.5, 1.5])  # ordered thresholds for 5 categories

    # Extended thresholds with -inf and +inf
    taus_ext = np.concatenate(([-np.inf], taus, [np.inf]))

    # Compute probabilities for each response category
    probs = normal_ogive(taus_ext, a, theta)

    # Sanity check: rows sum to ~1
    assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-6)

    # Legend labels
    labels = [f"Response {c}" for c in range(1, probs.shape[0] + 1)]

    # Plot colors
    colors = ["#b5c7e7", "#8eaadb", "#698ecf", "#4472c4", "#335b9c"]

    # First plot axis
    ax_1 = f.add_subplot(gs_01[0, 0])

    # Initialize counter
    counter = 0

    # Cycle over response categories
    for c in range(probs.shape[0]):
        ax_1.plot(theta, probs[c], label=labels[c], color=colors[c])
        ax_1.set_xlabel(r"Latent trait $\theta$")
        ax_1.set_ylabel("Response probability")
        ax_1.set_title("Normal-ogive model")
        ax_1.set_ylim(0, 1.02)
        sns.despine()
        counter += 1

    # Plot legend
    legend_y = -0.25
    ax_1.legend(loc="upper left", bbox_to_anchor=(0, legend_y), ncol=2)

    # Plot cumulative distribution function

    # Example items
    items = [
        {"a": 0.5, "b": 0.0, "label": r"Low discrimination ($a=0.5$)"},
        {"a": 1.0, "b": 0.0, "label": r"Medium discrimination ($a=1$)"},
        {"a": 2.0, "b": 0.0, "label": r"High discrimination ($a=2$)"},
        {"a": 1.0, "b": -1.0, "label": r"Easier ($b=-1$)"},
        {"a": 1.0, "b": 1.0, "label": r"Harder ($b=1$)"},
    ]

    # Plot colors
    colors = ["#698ecf", "#4472c4", "#335b9c", "#52bf90", "#317256"]

    # Second plot axis
    ax_2 = f.add_subplot(gs_01[0, 1])

    # Reset counter
    counter = 0

    # Plot zero line
    ax_2.axvline(0, color="gray", linestyle="--", alpha=0.5)

    # Cycle over response categories
    for c in items:
        p = norm.cdf(c["a"] * (theta - c["b"]))
        ax_2.plot(theta, p, color=colors[counter], label=c["label"])
        ax_2.set_xlabel(r"Latent trait $\theta$")
        ax_2.set_ylabel(r"Probability")
        ax_2.set_title("Gaussian cumulative distribution function")
        sns.despine()
        counter += 1

    # Plot legend
    ax_2.legend(loc="upper left", bbox_to_anchor=(0, legend_y), ncol=1)

    # ----------------------------------
    # Add subplot labels and save figure
    # ----------------------------------

    texts = ["a", "b"]  # label letters
    label_subplots(f, texts, x_offset=0.07, y_offset=0.01)

    # Save figure
    save_name = (
        "/"
        + home_dir
        + "/rasmus/Dropbox/Apps/Overleaf/FOR-1-Paper/Figures/normal_ogive.pdf"
    )
    plt.savefig(save_name, dpi=400)

    plt.ioff()
    plt.show()
