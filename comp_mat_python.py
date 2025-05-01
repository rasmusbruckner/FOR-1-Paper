import pandas as pd
from recovery_summary import recovery_summary
import matplotlib.pyplot as plt


parameters_mat = pd.read_csv('/home/rasmus/Dropbox/Hamburg Post-Doc/FOR_1_Paper/for_analysisPipeline/parameters.csv')

parameters_python = pd.read_pickle('/home/rasmus/Dropbox/for_analyses/results_df.pkl')

a = 1

behav_labels = [
    "beta_0", "beta_1", "beta_2", "beta_3",
    "beta_4", "beta_5",
    "omikron_0", "omikron_1"
]  # "beta_6", "beta_7"

# Filter based on estimated parameters
#which_params_vec = list(reg_vars.which_vars.values())
#behav_labels = [label for label, use in zip(behav_labels, which_params_vec) if use]

grid_size = (3, 3)

recovery_summary(parameters_mat, parameters_python, behav_labels, grid_size)

#plt.ioff()
plt.show()
a = 1