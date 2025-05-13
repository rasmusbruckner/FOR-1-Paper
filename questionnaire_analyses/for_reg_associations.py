import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from all_in import latex_plt

from FOR_1_Paper.for_utilities import (plot_idas_correlations,
                                       plot_main_questionnaire_correlations,
                                       plot_questionnaire_correlations_noise)

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Use preferred backend for Linux, or just take default
try:
    matplotlib.use("Qt5Agg")
except ImportError:
    pass

# todo: correlation ius and spq?

# Todo: in preprocessing
drop_ind = 70

# Questionnaires
df_questionnaires = pd.read_pickle("for_data/questionnaires_totalscores.pkl")
df_questionnaires = df_questionnaires.iloc[:-2]  # todo: fix this in preprocessing
df_questionnaires = df_questionnaires.sort_values(by=["subj_num"])
df_questionnaires = df_questionnaires.drop(drop_ind)

# ok es ist echt super wichtig, dass ich sichergehe, dass subs gematcht sind bei regression und questionnaires...
df_questionnaires = df_questionnaires.reset_index(drop=True)
df_questionnaires["gender"] = df_questionnaires["gender"].replace({1: -1, 2: 1})

# Load data of best regression model
df_regression = pd.read_pickle("for_data/regression_model_2_3_50_sp.pkl")

# Load data for regression applied separately for low and high noise
df_regression_low_noise = pd.read_pickle(
    "for_data/regression_model_low_noise_2_3_50_sp.pkl"
)
df_regression_high_noise = pd.read_pickle(
    "for_data/regression_model_high_noise_2_3_50_sp.pkl"
)

# Sort values just to be sure
df_regression = df_regression.sort_values(by=["subj_num"])
df_regression_low_noise = df_regression_low_noise.sort_values(by=["subj_num"])
df_regression_high_noise = df_regression_high_noise.sort_values(by=["subj_num"])
df_regression = df_regression.drop(drop_ind)
df_regression = df_regression.reset_index(drop=True)
df_regression_low_noise = df_regression_low_noise.drop(drop_ind).reset_index(drop=True)
df_regression_high_noise = df_regression_high_noise.drop(drop_ind).reset_index(
    drop=True
)

# -----------------
# Plot correlations
# -----------------

# Turn interactive mode on for plotting in debugger on Linux
plt.ion()

# ---------------------
# 1. Plot CAPE, IUS, SPQ
# ---------------------

# todo: die optionalen korrelationen ausprobieren...

# Fixed learning rate
# -------------------

var = "beta_1"
ylabel = "Fixed Learning Rate"
use_corr = False
plot_main_questionnaire_correlations(
    df_questionnaires, df_regression, var, ylabel, use_corr=use_corr
)
savename = "figures/main_questionnaires_fixedLR.png"
plt.savefig(savename, transparent=False, dpi=400)

# Adaptive learning rate
# ----------------------

var = "beta_4"
ylabel = "Adaptive Learning Rate"
plot_main_questionnaire_correlations(
    df_questionnaires, df_regression, var, ylabel, use_corr=use_corr
)
savename = "figures/main_questionnaires_adaptiveLR.png"
plt.savefig(savename, transparent=False, dpi=400)

# Fixed learning rate high vs low noise
# -------------------------------------

var = "beta_1"
ylabel = "Fixed Learning Rate"
plot_questionnaire_correlations_noise(
    df_questionnaires,
    df_regression_low_noise,
    df_regression_high_noise,
    var,
    ylabel,
    use_corr=use_corr,
)
savename = "figures/IUS_SPQ_low_high_noise_fixedLR.png"
plt.savefig(savename, transparent=False, dpi=400)

# Adaptive learning rate high vs low noise
# ----------------------------------------

var = "beta_4"
ylabel = "Adaptive Learning Rate"
plot_questionnaire_correlations_noise(
    df_questionnaires,
    df_regression_low_noise,
    df_regression_high_noise,
    var,
    ylabel,
    use_corr=use_corr,
)
savename = "figures/IUS_SPQ_low_high_noise_adaptiveLR.png"
plt.savefig(savename, transparent=False, dpi=400)

# -----------------
# 2. IDAS subscales
# -----------------

# Fixed learning rate
# -------------------

var = "beta_1"
ylabel = "Fixed Learning Rate"
plot_idas_correlations(df_questionnaires, df_regression, var, ylabel, use_corr=use_corr)
savename = "figures/IDAS_fixedLR.png"
plt.savefig(savename, transparent=False, dpi=400)

# Adaptive learning rate
# ----------------------

var = "beta_4"
ylabel = "Adaptive Learning Rate"
plot_idas_correlations(df_questionnaires, df_regression, var, ylabel, use_corr=use_corr)
savename = "figures/IDAS_adaptiveLR.png"
plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.ioff()
plt.show()
