if __name__ == "__main__":

    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    from all_in import latex_plt
    from rbm_analyses import parameter_summary

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    # Use preferred backend for Linux, or just take default
    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        pass

    # Load data
    model_2 = pd.read_pickle("for_data/regression_model_2_3_50_sp.pkl")

    behav_labels = [
        "beta_0",
        "beta_1",
        # "beta_2",
        # "beta_3",
        "beta_4",
        "beta_5",
        "beta_6",
        "omikron_0",
        "omikron_1",
    ]

    grid_size = (3, 3)
    parameter_summary(model_2, behav_labels, grid_size)

    # Show plot
    plt.ioff()
    plt.show()
