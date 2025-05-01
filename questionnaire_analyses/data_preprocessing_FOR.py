import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
#from sklearn.preprocessing import StandardScaler
from scipy import stats
from FOR_1_Paper.for_utilities import safe_save_dataframe

# predefine list of questionnaire_analyses used in the project
qns_list = ['CAPE1', 'CAPE2', 'SPQ1', 'IDAS1', 'IDAS2', 'IUS1']

# import data

#qns_data = pd.read_csv('../Data/QuestionnaireData/results_survey.csv')
qns_data = pd.read_csv('for_data/results_survey.csv')

qns_data.head()

# Drop 'nan' values from subjects
qns_data = qns_data.dropna(subset=['code'])

# sort by code
qns_data = qns_data.sort_values(by=['code'])

# Extract unique subjects
subjects = qns_data['code'].unique()
print("Number of subjects = {0}".format(len(subjects)))

# Create a mapping dictionary for renaming columns
column_mapping = {}

for col in qns_data.columns:
    if "[2]" in col and "CAPE" in col:
        # If the column contains [2], rename it to CAPE2A or CAPE2B based on the original name
        original_name = col.split("[2]")[0]
        without_CAPE = original_name.split("CAPE1")[1]
        new_name = f"CAPE2{without_CAPE}"
        column_mapping[col] = new_name
    else:
        # Otherwise, keep the original nam
        column_mapping[col] = col

# Rename the columns using the mapping dictionary
qns_data = qns_data.rename(columns=column_mapping)

# Correct CAPE2 encoding (for values where CAPE1 is 0, CAPE2 should be nan)
# Loop through columns
for col in qns_data.columns:
    if 'CAPE1' in col:
        identifier = col.split('[')[1].split(']')[0]

        cape1_col = col
        cape2_col = f'CAPE2[{identifier}]'

        # Set CAPE2 values to NaN where CAPE1 is 0
        qns_data.loc[qns_data[cape1_col] == 0, cape2_col] = np.nan

#qns_data.to_csv('../Data/QuestionnaireData/results_survey_CAPECorrected_nanCorrected_new.csv')
qns_data.to_csv('results_survey_CAPECorrected_nanCorrected_new.csv')

# Extract total scores for each subject for each different questionnaire
df_totalscore = pd.DataFrame(index=np.arange(0, len(subjects)), columns=qns_list)

id = list()
for subjInd, subj in enumerate(subjects):
    # enumerate through all questionnaire_analyses
    id.append(int(subj))

    for q_no, qname in enumerate(qns_list):
        df2 = qns_data[qns_data['code'] == subj].filter(regex=rf"^{qname}.*")
        # Convert all values to floats
        df2 = df2.astype(float)

        # Sum values for each questionnaire
        df_totalscore.iloc[subjInd, q_no] = float(df2.sum(axis=1).iloc[0])
        #df_totalscore.iloc[subjInd, q_no] = df2.sum(axis=1)


# Convert all columns of df_totalscore into float
df_totalscore = df_totalscore.astype(float)

df_totalscore['ID'] = id

# save df_totalscore
#df_totalscore.to_csv('../Data/QuestionnaireData/questionnaires_totalscores.csv', sep=';', index=False)
df_totalscore.to_csv('questionnaires_totalscores.csv', sep=';', index=False)

df_totalscore.name = 'questionnaires_totalscores'
safe_save_dataframe(df_totalscore)

# plot histograms of scores across all participants
# df_totalscore.hist(bins=12)Questionnaire

# Scatter plots of total scores along different questionnaire_analyses

# plt.scatter(df_totalscore['SPQ1'], df_totalscore['IDAS1'])
# plt.xlabel('SPQ1')
# plt.ylabel('IDAS1')
# plt.show()

# plot 3D plots using total scores
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(stats.zscore(df_totalscore['SPQ1']), stats.zscore(df_totalscore['IDAS2']),
             stats.zscore(df_totalscore['IUS1']))
plt.show()

# -----
# If the column doesn't contain [2], keep the original name
# remove [1] from the end
# if "[1]" in col:
#     new_name = col.split("[1]")[0]
#     column_mapping[col] = new_name
# else:
#     # If the column doesn't contain [1], keep the original name
#     # remove [2] from the end
#     if "[2]" in col:
#         new_name = col.split("[2]")[0]
#         column_mapping[col] = new_name
#     else:
#         # If the column doesn't contain [1] or [2], keep the original name
#         # remove [3] from the end
#         if "[3]" in col:
#             new_name = col.split("[3]")[0]
#             column_mapping[col] = new_name
#         else:
# Keep the original name if no brackets are present
