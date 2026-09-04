"""Figure 1: Illustration of the SCA framework.

Todo: Ensure colors are consistent across figures.
"""

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
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from allinpy import cm2inch, label_subplots, latex_plt

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Turn on interactive mode
plt.ion()

# Get home directory
paths = os.getcwd()
path = paths.split(os.path.sep)
home_dir = path[1]

# Control random seed for reproducibility
np.random.seed(42)

# --------------
# Prepare figure
# --------------

# Figure size
fig_width = 15
fig_height = 6

# Font size
font_size = 6

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create subplot grid and axis
gs_0 = gridspec.GridSpec(1, 3, left=0.0, right=0.99, bottom=0.1, wspace=0.4)
gs_01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[0, 0:2], wspace=0.05)
ax_a = f.add_subplot(gs_01[0, 0])
ax_b = f.add_subplot(gs_01[0, 1])
gs_02 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[0, 2])
ax_c = f.add_subplot(gs_02[0, 0])

# Consistent blue for this figure (todo: ensure consistence across figures)
blue = "#1f77b4"

# ---------------------
# Plot the four pillars
# ---------------------

# Pillar dimensions and positions
pillars = [
    "Reliability",
    "Construct Validity",
    "Generalizability",
    "Transparency",
]
pillar_w, pillar_h, pillar_y = 0.12, 0.4, 0.2
start_x = 0.14
p_gap = (0.72 - (4 * pillar_w)) / 3

# Cycle over pillars
for i, name in enumerate(pillars):
    pillar_x = start_x + i * (pillar_w + p_gap)
    transparency_pillar = "Transparency" in name

    # Clean, balanced column representations
    rect = patches.Rectangle(
        (pillar_x, pillar_y),
        pillar_w,
        pillar_h,
        edgecolor="#000000",
        facecolor=blue if transparency_pillar else "#FAFAFA",
        alpha=0.6 if transparency_pillar else 1,
        linewidth=1.2,
    )
    ax_a.add_patch(rect)

    # Capital and base architectural accents
    ax_a.plot(
        [pillar_x - 0.02, pillar_x + pillar_w + 0.02],
        [pillar_y + pillar_h, pillar_y + pillar_h],
        color="#000000",
        linewidth=1.4,
    )
    ax_a.plot(
        [pillar_x - 0.02, pillar_x + pillar_w + 0.02],
        [pillar_y, pillar_y],
        color="#000000",
        linewidth=1.4,
    )
    ax_a.text(
        pillar_x + pillar_w / 2,
        pillar_y + pillar_h / 2,
        name,
        ha="center",
        va="center",
        fontsize=font_size,
        weight="bold" if transparency_pillar else "normal",
        color="#000000",
        rotation=90,
    )

# --------------------------
# Plot the "temple" elements
# --------------------------

# Architrave
lw_frame = 1.4
architrave_y = 0.62
architrave_height = 0.07
architrave = patches.Rectangle(
    # (0.11, 0.61),
    (0.11, architrave_y),
    0.78,
    architrave_height,
    edgecolor="#000000",
    facecolor="#FFFFFF",
    linewidth=lw_frame,
)
ax_a.add_patch(architrave)

# Computational psychiatry text
ax_a.text(
    0.5,
    architrave_y + architrave_height / 2,
    "COMPUTATIONAL PSYCHIATRY",
    ha="center",
    va="center",
    fontsize=font_size,
    weight="bold",
)

# Roof
roof_y = architrave_y + architrave_height + 0.02
roof = patches.Polygon(
    [[0.08, roof_y], [0.5, 0.86], [0.92, roof_y]],
    closed=True,
    edgecolor="#000000",
    facecolor="#EBEBEB",
    linewidth=lw_frame,
)
ax_a.add_patch(roof)

# Base stylobate
base_stylobate = patches.Rectangle(
    (0.06, 0.13),
    0.88,
    0.05,
    edgecolor="#000000",
    facecolor="#EBEBEB",
    linewidth=lw_frame,
)
ax_a.add_patch(base_stylobate)

# Delete unnecessary axes
ax_a.axis("off")

# ------------------------------------------------------
# Plot the set of possible and reasonable specifications
# ------------------------------------------------------

# Y-position of the ellipses
ellipse_y = pillar_h
ellipse_width = 0.9
ellipse_height = 0.6

# Outer ellipse
outer_ellipse = patches.Ellipse(
    (0.5, ellipse_y),
    ellipse_width,
    ellipse_height,
    edgecolor="#000000",
    facecolor="none",
    linewidth=1.5,
)
ax_b.add_patch(outer_ellipse)

# Update ellipse size
ellipse_width *= 2 / 3
ellipse_height *= 2 / 3

# Inner ellipse
inner_ellipse = patches.Ellipse(
    (0.5, ellipse_y),
    ellipse_width,
    ellipse_height,
    edgecolor="#333333",
    facecolor="#E5E5E5",
    linewidth=1.2,
    zorder=1,
)
ax_b.add_patch(inner_ellipse)

# Box for all specifications
box_design = dict(
    boxstyle="round,pad=0.5",
    facecolor="white",
    alpha=0.7,
    edgecolor="k",
    linewidth=1,
)
ax_b.text(
    0.25,
    0.9,
    "All conceivable\nspecifications",
    ha="center",
    va="top",
    fontsize=font_size,
    bbox=box_design,
)

# Box for defensible specifications
box_design = dict(
    boxstyle="round,pad=0.5",
    facecolor="lightgray",
    alpha=0.7,
    edgecolor="k",
    linewidth=1,
)
ax_b.text(
    0.75,
    0.9,
    "Defensible\nspecifications",
    ha="center",
    va="top",
    fontsize=font_size,
    bbox=box_design,
)

# Randomly generate points within the inner ellipse
n_specs = 75  # number of example specifications
x_clean = np.random.normal(loc=0.5, scale=0.11, size=n_specs)
spec_offset = ellipse_height / 2
y_clean = np.random.normal(loc=ellipse_y, scale=0.07, size=n_specs)

# Clip coordinates strictly inside the visual bounds of the inner ellipse
distance = ((x_clean - 0.5) / 0.28) ** 2 + ((y_clean - ellipse_y) / 0.18) ** 2
valid_mask = distance < 0.95
ax_b.scatter(
    x_clean[valid_mask],
    y_clean[valid_mask],
    color=blue,
    alpha=1,
    s=10,
    edgecolors="none",
    zorder=2,
)

ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 1)
ax_b.axis("off")

# ----------------------------------------------------------
# Generate random specifications and corresponding estimates
# ----------------------------------------------------------

# Generate random specifications
pos_effects_int = int(n_specs * 0.85)
estimates = np.concatenate(
    [
        np.random.normal(loc=0.34, scale=0.05, size=pos_effects_int),
        np.random.normal(loc=-0.14, scale=0.07, size=n_specs - pos_effects_int),
    ]
)
estimates = np.clip(estimates, -0.32, 0.76)
sorted_estimates = np.sort(estimates)

# Specification curve
ax_c.plot(range(n_specs), sorted_estimates, color=blue, zorder=2)

# Baseline reference line at zero
ax_c.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=1)

# Axis constraints and neat labeling
ax_c.set_xlabel("Specification", labelpad=6)
ax_c.set_ylabel("Effect size", labelpad=4)
ax_c.set_xlim(-4, n_specs + 4)
ax_c.set_ylim(-0.35, 0.8)
ax_c.set_yticks([-0.25, 0.00, 0.25, 0.50, 0.75])

# Adjust position
# ----------------

# Get the current bounding box coordinates
pos_c = ax_c.get_position()

# Move up
v_offset = 0.1  # vertical offset
new_bottom = pos_c.y0 + v_offset
ax_c.set_position([pos_c.x0, new_bottom, pos_c.width, pos_c.height])

# Despine
sns.despine()

# Add labels
texts = ["a", "b", "c"]
x_offset = [-0.01, -0.01, 0.09]
y_offset = [0.05, 0.05, -0.05]
label_subplots(f, texts, x_offset=x_offset, y_offset=y_offset)

# Save figure
save_name = (
    "/"
    + home_dir
    + "/rasmus/Dropbox/Apps/Overleaf/FOR-1-Paper/Figures/for_figure_1.pdf"
)
plt.savefig(save_name, dpi=400)

# Show plot
plt.ioff()
plt.show()
