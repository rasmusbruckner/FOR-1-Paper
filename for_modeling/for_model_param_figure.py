"""Simple plot of parameter estimates."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from all_in import latex_plt
from ForEstVars import ForEstVars
from rbm_analyses import parameter_summary

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

# Load data
model = pd.read_pickle("for_data/for_estimates_10_sp.pkl")

# Call AlEstVars object
est_vars = ForEstVars()

# Free parameters
est_vars.which_vars = {
    est_vars.omikron_0: True,  # motor noise
    est_vars.omikron_1: True,  # learning-rate noise
    est_vars.h: True,  # hazard rate
    est_vars.s: True,  # surprise sensitivity
    est_vars.u: True,  # uncertainty underestimation
    est_vars.sigma_H: True,  # catch trials
}

# Plot results
# ------------

behav_labels = [
    "omikron_0",
    "omikron_1",
    "h",
    "s",
    "u",
    "sigma_H",
]

# Filter based on estimated parameters
which_params_vec = list(est_vars.which_vars.values())
behav_labels = [label for label, use in zip(behav_labels, which_params_vec) if use]

grid_size = (2, 3)
parameter_summary(model, behav_labels, grid_size)

# Show plot
plt.ioff()
plt.show()
