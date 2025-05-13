import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from all_in import latex_plt

from FOR_1_Paper.for_utilities import (plot_main_questionnaire_correlations,
                                       plot_questionnaire_correlations_noise)

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Use preferred backend for Linux, or just take default
try:
    matplotlib.use("Qt5Agg")
except ImportError:
    pass

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

# Load data excluding change points for estimation errors
df_for = pd.read_pickle("for_data/data_prepr_model.pkl")
no_cp = df_for["c_t"].to_numpy() == 0
df_for = df_for[no_cp]

# Compute estimation error for each subject
mean_per_subject = df_for.groupby("subj_num")["e_t"].mean().reset_index()
mean_per_subject = mean_per_subject.drop(drop_ind)
mean_per_subject = mean_per_subject.reset_index(drop=True)

# Compute estimation error for noise conditions
mean_per_subject_noise = (
    df_for.groupby(["subj_num", "kappa_t"])["e_t"].mean().reset_index()
)
mean_per_subject_low_noise = mean_per_subject_noise[
    mean_per_subject_noise["kappa_t"] == 16
].reset_index(drop=True)
mean_per_subject_high_noise = mean_per_subject_noise[
    mean_per_subject_noise["kappa_t"] == 8
].reset_index(drop=True)
mean_per_subject_low_noise = mean_per_subject_low_noise.drop(drop_ind)
mean_per_subject_high_noise = mean_per_subject_high_noise.drop(drop_ind)
mean_per_subject_low_noise = mean_per_subject_low_noise.reset_index(drop=True)
mean_per_subject_high_noise = mean_per_subject_high_noise.reset_index(drop=True)

# -----------------
# Plot correlations
# -----------------

# Turn interactive mode on for plotting in debugger on Linux
plt.ion()

# 1. Plot CAPE, IUS, SPQ
# ---------------------

var = "e_t"
ylabel = "Estimation Error"
use_corr = False
plot_main_questionnaire_correlations(
    df_questionnaires, mean_per_subject, var, ylabel, use_corr=use_corr
)
savename = "figures/main_questionnaires_est_err.png"
plt.savefig(savename, transparent=False, dpi=400)

# 2. Fixed learning rate high vs low noise
# ----------------------------------------

plot_questionnaire_correlations_noise(
    df_questionnaires,
    mean_per_subject_low_noise,
    mean_per_subject_high_noise,
    var,
    ylabel,
    use_corr=use_corr,
)
savename = "figures/IUS_SPQ_low_high_noise_est_err.png"
plt.savefig(savename, transparent=False, dpi=400)

# Show plot
plt.ioff()
plt.show()
