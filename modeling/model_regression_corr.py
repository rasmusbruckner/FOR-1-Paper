"""Correlating regression parameters with RBM estimates."""

import os

import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from allinpy import latex_plt
from ForEstVars import ForEstVars
from rbmpy import parameter_summary
import seaborn as sns

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
model = pd.read_pickle("for_data/rbm_estimates_10sp.pkl")

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

# Load data
model_23 = pd.read_pickle("for_data/regression_23_50sp.pkl")

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
    "Fixed LR",
    "Adaptive LR",
    "Catch effect",
    "Noise condition",
    "Motor Noise",
    "Learning-Rate Noise",
]

grid_size = (3, 3)
parameter_summary(model_23, behav_labels, grid_size, axis_labels=axis_labels)

plt.figure()
plt.scatter(model["h"], model_23["beta_1"])
r, p = stats.pearsonr(model["h"], model_23["beta_1"])
plt.xlabel("Hazard Rate")
plt.ylabel("Fixed Learning Rate")
plt.title("r = " + str(np.round(r, 3)))

# Show plot
sns.despine()
plt.ioff()
plt.show()
