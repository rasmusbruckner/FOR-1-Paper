"""Simple plot of parameter estimates."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from allinpy import latex_plt, label_subplots
from ForEstVars import ForEstVars
from rbmpy import parameter_summary

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Use preferred backend for Linux, or just take default
try:
    matplotlib.use("Qt5Agg")
except ImportError:
    pass

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# Turn on interactive mode
plt.ion()

# ---------
# Load data
# ---------

# Load data
model = pd.read_pickle("for_data/rbm_estimates_10sp.pkl")

# Call AlEstVars object
est_vars = ForEstVars()

# ------------
# Plot results
# ------------

# Free parameters
est_vars.which_vars = {
    est_vars.omikron_0: True,  # motor noise
    est_vars.omikron_1: True,  # learning-rate noise
    est_vars.lambda_0: False,  # no perseveration
    est_vars.lambda_1: False,  # no perseveration
    est_vars.h: True,  # hazard rate
    est_vars.s: True,  # surprise sensitivity
    est_vars.u: True,  # uncertainty underestimation
    est_vars.sigma_H: True,  # catch trials
}

behav_labels = [
    "omikron_0",
    "omikron_1",
    "lambda_0",
    "lambda_1",
    "h",
    "s",
    "u",
    "sigma_H",
]

axis_labels = [
        "Motor noise",
        "Learning-rate noise",
        "Hazard rate",
        "Surprise sensitivity",
        "Uncertainty underestimation",
        "Catch trials"
    ]

# Filter based on estimated parameters
which_params_vec = list(est_vars.which_vars.values())
behav_labels = [label for label, use in zip(behav_labels, which_params_vec) if use]

grid_size = (2, 3)
f = parameter_summary(model, behav_labels, grid_size, axis_labels=axis_labels)

# ----------------------------------
# Add subplot labels and save figure
# ----------------------------------

texts = ["a", "b", "c", "d", "e", "f", "g", "h"]  # label letters
label_subplots(f, texts, 0.07, y_offset=0.01)

# Save figure
save_name = (
        "/"
        + home_dir
        + "/rasmus/Dropbox/Apps/Overleaf/FOR-1-Paper/Figures/model_param_figure.pdf"
)
plt.savefig(save_name, dpi=400)

# Show plot
plt.ioff()
plt.show()
