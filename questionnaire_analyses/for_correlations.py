import numpy as np
import pandas as pd
import scipy.stats as stats

# Regression
df_regression = pd.read_pickle("for_data/regression_15_sp.pkl")

df_regression = df_regression.sort_values(by=['ID'])


# Questionnaires
df_questionnaires = pd.read_pickle("for_data/questionnaires_totalscores.pkl")
new_df = df_questionnaires.iloc[:-2]

new_df = new_df.sort_values(by=['ID'])

import matplotlib.pyplot as plt
from all_in import cm2inch
import seaborn as sns

print('CAPE1 all:', np.corrcoef(df_regression['beta_1'][:], new_df['CAPE1'][:]))
print('CAPE1 32:', np.corrcoef(df_regression['beta_1'][:32], new_df['CAPE1'][:32]))

a, b = stats.pearsonr(df_regression['beta_1'][:], new_df['CAPE1'][:])


fig_width = 10
fig_height = 10
# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
plt.scatter(df_regression['beta_1'][:], new_df['CAPE1'][:], alpha=0.6, label='Predicted mean')

# Fit a line
x = df_regression['beta_1'][:]
y = new_df['CAPE1'][:]
slope, intercept = np.polyfit(x, y, 1)
plt.plot(x, slope * x + intercept, color='red', label='Linear fit')

#plt.title(f"Across Subjects: Correlation: r = {r:.3f}")
plt.xlabel("Fixed Learning Rate")
plt.ylabel("CAPE Score")

# Correlate with actual log RT
r, p = stats.pearsonr(x, y)
print(f"r = {p:.3f}")
plt.title(f"r = {r:.3f}; p = {p:.3f}")

plt.tight_layout()
sns.despine()

savename = "figures/cape_corr.png"
plt.savefig(savename, transparent=False, dpi=400)
plt.ioff()
plt.show()
#plt.close()


print('SPQ1 all', np.corrcoef(df_regression['beta_1'][:], new_df['SPQ1'][:]))
print('SPQ1 32:', np.corrcoef(df_regression['beta_1'][:32], new_df['SPQ1'][:32]))
a, b = stats.pearsonr(df_regression['beta_1'][:], new_df['SPQ1'][:])

print('IDAS1 all', np.corrcoef(df_regression['beta_1'][:], new_df['IDAS1'][:]))
print('IDAS1 32:', np.corrcoef(df_regression['beta_1'][:32], new_df['IDAS1'][:32]))
a, b = stats.pearsonr(df_regression['beta_1'][:], new_df['IDAS1'][:])

print('IDAS2 all', np.corrcoef(df_regression['beta_1'][:], new_df['IDAS2'][:]))
print('IDAS2 32:', np.corrcoef(df_regression['beta_1'][:32], new_df['IDAS2'][:32]))
a, b = stats.pearsonr(df_regression['beta_1'][:], new_df['IDAS2'][:])

fig_width = 10
fig_height = 10

# Fit a line
x = df_regression['beta_1'][:]
y = new_df['IDAS2'][:]

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
plt.scatter(x, y, alpha=0.6, label='Predicted mean')


slope, intercept = np.polyfit(x, y, 1)
plt.plot(x, slope * x + intercept, color='red', label='Linear fit')

#plt.title(f"Across Subjects: Correlation: r = {r:.3f}")
plt.xlabel("Fixed Learning Rate")
plt.ylabel("IDAS2 Score")

# Correlate with actual log RT
r, p = stats.pearsonr(x, y)
print(f"r = {p:.3f}")
plt.title(f"r = {r:.3f}; p = {p:.3f}")

plt.tight_layout()
sns.despine()

savename = "figures/idas2_corr.png"
plt.savefig(savename, transparent=False, dpi=400)
plt.ioff()
plt.show()





# Ensure subj_num is matched
a = 1

# Compute first correlations

# Compute correlations with 32 participants

