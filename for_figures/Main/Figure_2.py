"""Figure 2: Task and model

1. Load data
2. Prepare figure
3. Plot task trial schematic
4. Plot block example and model computations
5. Plot regression example
6. Add subplot labels and save figure

# Todo: ensure colors are consistent across figures
"""

import os
import platform

import matplotlib

system = platform.system()

# Simple cross-platform backend selection
if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")  # headless
elif platform.system() == "Darwin":
    matplotlib.use("MacOSX")  # macOS native
else:
    matplotlib.use("Qt5Agg")  # Linux with display, Windows, others

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from allinpy import cm2inch, label_subplots, latex_plt, plot_image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rbmpy import AgentVars, AlAgent

from FOR_1_Paper.modeling.simulation_rbm import simulation

# Create function to plot cannon illustration
# -------------------------------------------


def cannon_illustration(fig: Figure, ax: Axes, cannon_image_path) -> None:
    """Plots the cannon illustration.

    Parameters
    ----------
    fig : Figure
        Figure object.
    ax : Axes
        Axes object.
    cannon_image_path : list
        List of image paths.

    Returns
    -------
    None
        This function does not return any value.
    """

    # Figure text and font size
    text = ["Prediction", "Outcome", "Prediction\nError", "Shield"]
    fontsize = 6

    # Initialize image coordinates
    cell_x0 = -0.1
    cell_x1 = 0.2
    image_y = 0.8

    # Initialize text coordinates
    text_y_dist = [0.025, 0.025, 0.025, 0.025]
    text_pos = "left_below"

    # Cycle over images
    for j in range(0, 4):

        # Plot images and text
        plot_image(
            fig,
            cannon_image_path[j],
            cell_x0,
            cell_x1,
            image_y,
            ax_0,
            text_y_dist[j],
            text[j],
            text_pos,
            fontsize,
            zoom=0.025,
        )

        # Update coordinates
        cell_x0 += 0.25
        cell_x1 += 0.25
        image_y += -0.2

    # Delete unnecessary axes
    ax.axis("off")


# Turn on interactive mode
plt.ion()

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# ------------
# 1. Load data
# ------------

df_for = pd.read_pickle("for_data/data_prepr.pkl")
df_for = df_for.dropna(subset=["delta_t_rad", "a_t_rad"]).reset_index()  # drop nans

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_height = 10
fig_width = 15

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(
    3, 2, wspace=0.25, hspace=0.4, top=0.95, bottom=0.085, left=0.1, right=0.975
)

# Y-label distance
ylabel_dist = -0.15

# Colors
light_blue = "#3282B8"
dark_blue = "#0F4C75"

# ----------------------------
# 3. Plot task trial schematic
# ----------------------------

# Create subplot grid and axis
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[0, 0])
ax_0 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_0)

# Image paths
image_paths = [
    "for_figures/for_pract_pred.png",
    "for_figures/for_pract_outcome.png",
    "for_figures/for_pract_pe.png",
    "for_figures/for_pract_shield.png",
]

# Plot cannon illustration
cannon_illustration(f, ax_0, image_paths)

# Create subplot grid and axis
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[0, 1])
ax_0 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax_0)

# Image paths
image_paths = [
    "for_figures/for_pred.png",
    "for_figures/for_outcome.png",
    "for_figures/for_pe.png",
    "for_figures/for_shield.png",
]

# Plot cannon illustration
cannon_illustration(f, ax_0, image_paths)

# --------------------------------------------
# 4. Plot block example and model computations
# --------------------------------------------

# Create subplot grid
gs_01 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_0[1:3, 0], hspace=0.5)
ax_10 = plt.Subplot(f, gs_01[0:2, 0])
f.add_subplot(ax_10)

# Simulation parameters
n_sim = 1
model_params = pd.DataFrame(
    columns=[
        "omikron_0",
        "omikron_1",
        "lambda_0",
        "lambda_1",
        "b_0",
        "b_1",
        "h",
        "s",
        "u",
        "q",
        "sigma_H",
        "d",
        "subj_num",
        "age_group",
    ]
)
model_params.loc[0, "omikron_0"] = 0.01
model_params.loc[0, "omikron_1"] = 0
model_params.loc[0, "b_0"] = -30
model_params.loc[0, "b_1"] = -1.5
model_params.loc[0, "h"] = 0.1
model_params.loc[0, "s"] = 1
model_params.loc[0, "u"] = 0
model_params.loc[0, "q"] = 0
model_params.loc[0, "sigma_H"] = 0.01
model_params.loc[0, "d"] = 0.0
model_params.loc[0, "subj_num"] = 1.0
model_params.loc[0, "age_group"] = 0

# Normative model simulation
sim_pers = False  # no perseveration simulation
sim_est_err, df_data, true_params = simulation(
    df_for, model_params, n_sim, sim_pers, sim="agent"
)

# Indicate plot range and x-axis
plot_range = (200, 225)
x = np.linspace(0, plot_range[1] - plot_range[0] - 1, plot_range[1] - plot_range[0])

# Mean and outcomes
ax_10.plot(
    x,
    (np.array(np.rad2deg(df_for["mu_t_rad"][plot_range[0] : plot_range[1]]))),
    "--",
    x,
    np.array(np.rad2deg(df_for["x_t_rad"][plot_range[0] : plot_range[1]])),
    ".",
    color="k",
)

# Model predictions
ax_10.plot(
    x,
    (np.array(np.rad2deg(df_data["mu_t_rad"][plot_range[0] : plot_range[1]]))),
    "-",
    color="r",
)

# Adjust plot styling
ax_10.set_ylabel("Position")
ax_10.yaxis.set_label_coords(ylabel_dist, 0.5)
ax_10.legend(["Cannon", "Outcome", "Model"], loc=1, framealpha=0.8)
ax_10.set_ylim(150, 369)

# Remove tick parameters
ax_10.tick_params(
    axis="x", labelbottom=False  # changes apply to the x-axis
)  # labels along the bottom edge are off

# Prediction errors
ax_11 = plt.Subplot(f, gs_01[2, 0])
f.add_subplot(ax_11)
ax_11.axhline(0, linestyle="--", color="gray")
ax_11.plot(
    x,
    np.array(np.rad2deg(df_data["delta_t_rad"][plot_range[0] : plot_range[1]])),
    linewidth=2,
    color="k",
    alpha=1,
)
ax_11.set_ylabel("Prediction Error")
ax_11.yaxis.set_label_coords(ylabel_dist, 0.5)

# Remove tick parameters
ax_11.tick_params(
    axis="x", labelbottom=False  # changes apply to the x-axis
)  # labels along the bottom edge are off

# Relative uncertainty, changepoint probability, and learning rate
ax_12 = plt.Subplot(f, gs_01[3, 0])
f.add_subplot(ax_12)
ax_12.plot(
    x,
    np.array(df_data["tau_t"][plot_range[0] : plot_range[1]]),
    linewidth=2,
    color=light_blue,
    alpha=1,
)
ax_12.plot(
    x,
    np.array(df_data["omega_t"][plot_range[0] : plot_range[1]]),
    linewidth=2,
    color=dark_blue,
    alpha=1,
)
ax_12.plot(
    x,
    np.array(df_data["alpha_t"][plot_range[0] : plot_range[1]]),
    linewidth=2,
    color="k",
    alpha=1,
)
ax_12.legend(["RU", "CPP", "LR"], loc=4)
ax_12.set_xlabel("Trial")
ax_12.set_ylabel("Variable")
ax_12.yaxis.set_label_coords(ylabel_dist, 0.5)

# --------------------------
# 5. Plot regression example
# --------------------------

# Create subplot grid
gs_01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[1:3, 1], hspace=0.5)
ax_10 = plt.Subplot(f, gs_01[:])
f.add_subplot(ax_10)

df_example = df_for.loc[
    df_for["subj_num"] == 29.0
]  # 8 15, 16, 22, 23, 29 are alternative examples that would illustrate both types of learning rate well

# Agent object instance
agent_vars = AgentVars()
agent = AlAgent(agent_vars)

# Initialize arrays
pe = np.linspace(-180, 180, 361)
alpha = np.full(361, np.nan)

# Set agent variables
agent_vars.h = 0.1
agent_vars.s = 1
agent_vars.u = np.exp(0)
agent_vars.q = 0
agent_vars.sigma_H = 0
agent_vars.circular = True
agent_vars.max_x = 2 * np.pi
agent_vars.mu_0 = 0
agent_vars.sigma = 0.25
agent_vars.sigma_0 = 0.1

# Cycle over prediction-error range
for i in range(len(pe)):

    agent = AlAgent(agent_vars)
    agent.tau_t = 0.2
    agent.learn(np.deg2rad(pe[i]), np.nan, False, np.nan, False)
    alpha[i] = agent.alpha_t

# Regression plot with learning rates
ax_10.axhline(0, linestyle="--", color="gray")
ax_10.axvline(0, linestyle="--", color="gray")
(a,) = ax_10.plot(
    df_example["delta_t_rad"], df_example["a_t_rad"], ".", color="gray", alpha=1
)
(b,) = ax_10.plot(np.deg2rad(pe), np.deg2rad(pe) * alpha, "-", color="k")
(c,) = ax_10.plot(np.deg2rad(pe), np.deg2rad(pe) * 0.8, "-", color=light_blue)
ax_10.legend(
    (c, b, a), ("Fixed LR", "Adaptive LR", "Example data"), loc=0, framealpha=0.8
)
ax_10.set_xlabel("Prediction Error")
ax_10.set_ylabel("Update")

# Delete unnecessary axes
sns.despine()

# -------------------------------------
# 6. Add subplot labels and save figure
# -------------------------------------

# Label letters
texts = ["a", "b", "c", "", "", "d"]

# Add labels
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

# Save figure
# -----------

# Save figure
save_name = (
    "/"
    + home_dir
    + "/rasmus/Dropbox/Apps/Overleaf/FOR-1-Paper/Figures/for_figure_2.pdf"
)
plt.savefig(save_name, dpi=400)

# Show plot
plt.ioff()
plt.show()
