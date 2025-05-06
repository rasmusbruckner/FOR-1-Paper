import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from all_in import cm2inch, latex_plt

from FOR_1_Paper.for_utilities import plot_questionnaire_correlation

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Use preferred backend for Linux, or just take default
try:
    matplotlib.use("Qt5Agg")
except ImportError:
    pass

# Todo:
#   1 identify potential outliers
#   2 check inferential stats Hashim
#   3 apply FDR correction
#   4 Bayes factors

# Regression
df_regression = pd.read_pickle("for_data/regression_10_sp.pkl")
df_regression = df_regression.sort_values(by=["subj_num"])

# Questionnaires
df_questionnaires = pd.read_pickle("for_data/questionnaires_totalscores.pkl")
df_questionnaires = df_questionnaires.iloc[:-2]  # todo: fix this in preprocessing
df_questionnaires = df_questionnaires.sort_values(by=["subj_num"])

# Turn interactive mode on for plotting in debugger on Linux
plt.ion()

# Plot CAPE sum score
# -------------------

fig_width = 10
fig_height = 10

# Select data
x = df_regression["beta_1"][:]
xlabel = "Fixed Learning Rate"
y = df_questionnaires["CAPE1"][:]
ylabel = "CAPE Score"

# Create and save figure
plt.figure(figsize=cm2inch(fig_width, fig_height))
plot_questionnaire_correlation(x, y, xlabel, ylabel)
savename = "figures/cape_corr.png"
plt.savefig(savename, transparent=False, dpi=400)

# Plot CAPE positive score
# ------------------------

# Select data
x = df_regression["beta_1"][:]
y = df_questionnaires["CAPE_pos"][:]
ylabel = "CAPE pos Score"

# Create and save figure
plt.figure(figsize=cm2inch(fig_width, fig_height))
plot_questionnaire_correlation(x, y, xlabel, ylabel)
savename = "figures/cape_pos_corr.png"
plt.savefig(savename, transparent=False, dpi=400)

# Plot CAPE negative score
# ------------------------

# Select data
x = df_regression["beta_1"][:]
y = df_questionnaires["CAPE_neg"][:]
ylabel = "CAPE neg Score"

# Create and save figure
plt.figure(figsize=cm2inch(fig_width, fig_height))
plot_questionnaire_correlation(x, y, xlabel, ylabel)
savename = "figures/cape_neg_corr.png"
plt.savefig(savename, transparent=False, dpi=400)

# Plot CAPE depression score
# --------------------------

# Select data
x = df_regression["beta_1"][:]
y = df_questionnaires["CAPE_dep"][:]
ylabel = "CAPE dep Score"

# Create and save figure
plt.figure(figsize=cm2inch(fig_width, fig_height))
plot_questionnaire_correlation(x, y, xlabel, ylabel)
savename = "figures/cape_dep_corr.png"
plt.savefig(savename, transparent=False, dpi=400)

# Plot IUS sum score
# ------------------

# Select data
x = df_regression["beta_1"][:]
y = df_questionnaires["IUS1"][:]
ylabel = "IUS Score"

# Create and save figure
plt.figure(figsize=cm2inch(fig_width, fig_height))
plot_questionnaire_correlation(x, y, xlabel, ylabel)
savename = "figures/IUS.png"
plt.savefig(savename, transparent=False, dpi=400)

# Plot SPQ sum score
# ------------------

# Select data
x = df_regression["beta_1"][:]
y = df_questionnaires["SPQ1"][:]
ylabel = "SPQ1 Score"

# Create and save figure
plt.figure(figsize=cm2inch(fig_width, fig_height))
plot_questionnaire_correlation(x, y, xlabel, ylabel)
savename = "figures/SPQ.png"
plt.savefig(savename, transparent=False, dpi=400)

# Plot IDAS sub scales
# --------------------

# Increase figure size
fig_width = 15
fig_height = 20

# Dysphoria
x = df_regression["beta_1"][:]
y = df_questionnaires["dysphoria"][:]
ylabel = "Dysphoria Score"

# Create and save figure
plt.figure(figsize=cm2inch(fig_width, fig_height))
plt.subplot(541)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Fatigue
y = df_questionnaires["fatigue"][:]
ylabel = "Fatigue Score"
plt.subplot(542)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Insomnia
y = df_questionnaires["insomnia"][:]
ylabel = "Insomnia Score"
plt.subplot(543)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Suicidality
y = df_questionnaires["suicidality"][:]
ylabel = "Suicidality Score"
plt.subplot(544)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Increase in appetite
y = df_questionnaires["incr_appetite"][:]
ylabel = "Increase Appetite Score"
plt.subplot(545)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Loss of appetite
y = df_questionnaires["loss_appetite"][:]
ylabel = "Loss Appetite Score"
plt.subplot(546)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Wellbeing
y = df_questionnaires["wellbeing"][:]
ylabel = "Wellbeing Score"
plt.subplot(547)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Moodiness
y = df_questionnaires["moodiness"][:]
ylabel = "Moodiness Score"
plt.subplot(548)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Mania
y = df_questionnaires["mania"][:]
ylabel = "Mania Score"
plt.subplot(549)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Euphoria
y = df_questionnaires["euphoria"][:]
ylabel = "Euphoria Score"
plt.subplot(5, 4, 10)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Social anxiety
y = df_questionnaires["social_anx"][:]
ylabel = "Social Anxiety Score"
plt.subplot(5, 4, 11)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Claustrophobia
y = df_questionnaires["claustrophobia"][:]
ylabel = "Claustrophobia Score"
plt.subplot(5, 4, 12)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Traumatic instrusions
y = df_questionnaires["traumatic_intrusions"][:]
ylabel = "Traumatic Intrusions Score"
plt.subplot(5, 4, 13)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Traumatic avoidance
y = df_questionnaires["traumatic_avoidance"][:]
ylabel = "Traumatic Avoidance Score"
plt.subplot(5, 4, 14)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Compulsion order
y = df_questionnaires["compulsion_order"][:]
ylabel = "Compulsion Order Score"
plt.subplot(5, 4, 15)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Compulsion clean
y = df_questionnaires["compulsion_clean"][:]
ylabel = "Compulsion Clean Score"
plt.subplot(5, 4, 16)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Compulsion control
y = df_questionnaires["compulsion_control"][:]
ylabel = "Compulsion Control Score"
plt.subplot(5, 4, 17)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Panic
y = df_questionnaires["panic"][:]
ylabel = "Panic Score"
plt.subplot(5, 4, 18)
plot_questionnaire_correlation(x, y, xlabel, ylabel)

# Delete unnecessary axes
plt.tight_layout()

# Save figure
savename = "figures/IDAS.png"
plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.ioff()
plt.show()
