import sys

import numpy as np
import pandas as pd

from FOR_1_Paper.for_utilities import safe_save_dataframe

# todo IDAS
#    non-overlapping dimensions!

# todo
#   1. overview what questionnaires we have

# CAPE: The community assessment of psychotic experience
#   todo: what are the subscales?
#       what is difference between cape1 and 2?

# SPQ: Schizotypal personality questionnaire
#   todo: why SPQ1 and what about other numbers (if that exists)

# IDAS: Inventory of anxiety and depression
#   todo: what is IDAS1 and 2

# IUS: Intolerance of uncertainty scale
#   todo: any subscale?

# Todo For all cases: ensure we can rely on simple sum scores

# Load data
qns_data = pd.read_csv("for_data/results_survey.csv")
qns_data = qns_data.rename(columns={"code": "subj_num"})

# Todo: why do we have nans and what are they doing here?
# Drop 'nan' values from subjects
qns_data = qns_data.dropna(subset=["subj_num"])

# Sort by subject number
qns_data = qns_data.sort_values(by=["subj_num"])
subjects = qns_data["subj_num"].unique()
n_subj = len(subjects)


# Drop unnecessary items
# todo: double check and what is psych?
qns_data.drop(
    [
        "id",
        "submitdate",
        "lastpage",
        "startlanguage",
        "seed",
        "startdate",
        "datestamp",
        "ipaddr",
        "psych[0]",
        "psych[1]",
        "psych[2]",
        "psych[3]",
        "psych[4]",
        "psych[5]",
        "psych[other]",
    ],
    axis=1,
    inplace=True,
)

# ---------------
# Preprocess CAPE
# ----------------

# This is probably dimension frequency (1) and distress (2)

# todo: wichtig: es gibt in CAPE1 nans... wie geht man damit um?
#   erstmal ohne nan summen gebildet, aber sicher schauen...
# Rename CAPE questions
# -------------------------
# Todo: why is this necessary?

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
        # Otherwise, keep the original name
        column_mapping[col] = col

# Rename the columns using the mapping dictionary
qns_data = qns_data.rename(columns=column_mapping)

# Todo: discuss with Hashim to better understand
# Correct CAPE2 encoding (for values where CAPE1 is 0, CAPE2 should be nan)
for col in qns_data.columns:
    if "CAPE1" in col:
        identifier = col.split("[")[1].split("]")[0]

        cape1_col = col
        cape2_col = f"CAPE2[{identifier}]"

        # Set CAPE2 values to NaN where CAPE1 is 0
        qns_data.loc[qns_data[cape1_col] == 0, cape2_col] = np.nan

# Save data
# Todo: safe_save and rename...
# todo: why here and not later?? just cape maybe...
# qns_data.to_csv('../Data/QuestionnaireData/results_survey_CAPECorrected_nanCorrected_new.csv')
# qns_data.to_csv("results_survey_CAPECorrected_nanCorrected_new.csv")

# Todo: what is this doing? also w/ cape?
# ----------------
# Total sum scores
# ----------------

# Questionnaires of interest
qns_list = [
    "CAPE1",
    "CAPE2",
    "SPQ1",
    "IUS1",
]  # "IDAS1", "IDAS2", todo: can't do with IDAS right?

# Initialize
subj_num = list()
age = list()
gender = list()

df_totalscore = pd.DataFrame(index=np.arange(0, n_subj), columns=qns_list)

# Cycle over subjects
for subjInd, subj in enumerate(subjects):

    # Store subject number
    subj_num.append(int(subj))
    age.append(qns_data[qns_data["subj_num"] == subj]['age'].values[0])
    gender.append(qns_data[qns_data["subj_num"] == subj]['sex'].values[0])

    # Cycle over all questionnaires
    for q_no, qname in enumerate(qns_list):

        # Extract current questionnaire
        df2 = (
            qns_data[qns_data["subj_num"] == subj]
            .filter(regex=rf"^{qname}.*")
            .astype(float)
        )

        # Sum values for each questionnaire
        df_totalscore.iloc[subjInd, q_no] = float(df2.sum(axis=1).iloc[0])

# Convert all columns of df_totalscore into float
df_totalscore = df_totalscore.astype(float)

# Add subject numbers
df_totalscore["subj_num"] = subj_num
df_totalscore["age"] = age
df_totalscore["gender"] = gender

# -----------------------------
# Extract CAPE dimension scores
# -----------------------------

# Positive: a2+a5+a6+a7+a10+a11+a13+a15+a17+a20+a22+a24+a26+a28+a30+a31+a33+a34+a41+a42
pos = [
    "CAPE1[CAPEP02][1]",
    "CAPE1[CAPEP05][1]",
    "CAPE1[CAPEP06][1]",
    "CAPE1[CAPEP07][1]",
    "CAPE1[CAPEP10][1]",
    "CAPE1[CAPEP11][1]",
    "CAPE1[CAPEP13][1]",
    "CAPE1[CAPEP15][1]",
    "CAPE1[CAPEP17][1]",
    "CAPE1[CAPEP20][1]",
    "CAPE1[CAPEP22][1]",
    "CAPE1[CAPEP24][1]",
    "CAPE1[CAPEP26][1]",
    "CAPE1[CAPEP28][1]",
    "CAPE1[CAPEP30][1]",
    "CAPE1[CAPEP31][1]",
    "CAPE1[CAPEP33][1]",
    "CAPE1[CAPEP34][1]",
    "CAPE1[CAPEP41][1]",
    "CAPE1[CAPEP42][1]",
]

# Negative: a3+a4+a8+a16+a18+a21+a23+a25+a27+a29+a32+a35+a36+a37
neg = [
    "CAPE1[CAPEP03][1]",
    "CAPE1[CAPEP04][1]",
    "CAPE1[CAPEP08][1]",
    "CAPE1[CAPEP16][1]",
    "CAPE1[CAPEP18][1]",
    "CAPE1[CAPEP21][1]",
    "CAPE1[CAPEP23][1]",
    "CAPE1[CAPEP25][1]",
    "CAPE1[CAPEP27][1]",
    "CAPE1[CAPEP29][1]",
    "CAPE1[CAPEP32][1]",
    "CAPE1[CAPEP35][1]",
    "CAPE1[CAPEP36][1]",
    "CAPE1[CAPEP37][1]",
]

# Depressive: a1+a9+a12+a14+a19+a38+a39+a40
dep = [
    "CAPE1[CAPEP01][1]",
    "CAPE1[CAPEP09][1]",
    "CAPE1[CAPEP12][1]",
    "CAPE1[CAPEP14][1]",
    "CAPE1[CAPEP19][1]",
    "CAPE1[CAPEP38][1]",
    "CAPE1[CAPEP39][1]",
    "CAPE1[CAPEP40][1]",
]

# Initialize dimension lists
pos_vals = list()
neg_vals = list()
dep_vals = list()

# Cycle over subjects
qname = "CAPE1"
for subjInd, subj in enumerate(subjects):

    # Re-initialize sum score variables and counter
    pos_sum = 0
    neg_sum = 0
    dep_sum = 0
    counter = 0

    # Select all questionnaire items of current subject
    df2 = qns_data[qns_data["subj_num"] == subj].filter(regex=rf"^{qname}.*")

    # Cycle over each question for current participant
    for col in df2.columns:

        # Only consider non-NaNs and update respective variables
        if not np.isnan(df2[col].iloc[0]):
            if col in pos:
                pos_sum += df2[col].iloc[0]
            elif col in neg:
                neg_sum += df2[col].iloc[0]
            elif col in dep:
                dep_sum += df2[col].iloc[0]
            else:
                sys.exit("delta_t is NaN")

        # Update counter
        counter += 1

    # Compare counter and total N of questions included
    if counter != len(pos) + len(neg) + len(dep):
        sys.exit("delta_t is NaN")
    else:
        # Add sum scores to lists
        pos_vals.append(pos_sum)
        neg_vals.append(neg_sum)
        dep_vals.append(dep_sum)

# Add dimension sum scores
df_totalscore["CAPE_pos"] = pos_vals
df_totalscore["CAPE_neg"] = neg_vals
df_totalscore["CAPE_dep"] = dep_vals

# -----------------------------
# Extract IDAS dimension scores
# -----------------------------

# 1) Dysphorie: 2, 5, 8, 9, 21, 31, 39, 47, 56, 60
dysphoria = [
    "IDAS1[IDAS02]",
    "IDAS1[IDAS05]",
    "IDAS1[IDAS08]",
    "IDAS1[IDAS09]",
    "IDAS1[IDAS21]",
    "IDAS1[IDAS31]",
    "IDAS1[IDAS39]",
    "IDAS2[IDAS47]",
    "IDAS2[IDAS56]",
    "IDAS2[IDAS60]",
]

# 2) Abgeschlagenheit: 6, 29, 30, 42, 53, 54
fatigue = [
    "IDAS1[IDAS06]",
    "IDAS1[IDAS29]",
    "IDAS1[IDAS30]",
    "IDAS1[IDAS42]",
    "IDAS2[IDAS53]",
    "IDAS2[IDAS54]",
]

# 3) Schlaflosigkeit: 4, 11, 17, 25, 36, 50
insomnia = [
    "IDAS1[IDAS04]",
    "IDAS1[IDAS11]",
    "IDAS1[IDAS17]",
    "IDAS1[IDAS25]",
    "IDAS1[IDAS36]",
    "IDAS2[IDAS50]",
]

# 4) Suizidalität: 13, 22, 33, 37, 45, 51
suicidality = [
    "IDAS1[IDAS13]",
    "IDAS1[IDAS22]",
    "IDAS1[IDAS33]",
    "IDAS1[IDAS37]",
    "IDAS2[IDAS45]",
    "IDAS2[IDAS51]",
]

# 5) Appetitsteigerung: 19, 24, 62
incr_appetite = ["IDAS1[IDAS19]", "IDAS1[IDAS24]", "IDAS2[IDAS62]"]

# 6) Appetitlosigkeit: 1, 26, 59
loss_appetite = ["IDAS1[IDAS01]", "IDAS1[IDAS26]", "IDAS2[IDAS59]"]

# 7) Wohlbefinden: 3, 10, 23, 27, 49, 52, 58, 63
wellbeing = [
    "IDAS1[IDAS03]",
    "IDAS1[IDAS10]",
    "IDAS1[IDAS23]",
    "IDAS1[IDAS27]",
    "IDAS2[IDAS49]",
    "IDAS2[IDAS52]",
    "IDAS2[IDAS58]",
    "IDAS2[IDAS63]",
]

# 8) Übellaunigkeit: 12, 35, 43, 61
moodiness = ["IDAS1[IDAS12]", "IDAS1[IDAS35]", "IDAS1[IDAS43]", "IDAS2[IDAS61]"]

# 9) Manie: 66, 70, 76, 82, 86
mania = [
    "IDAS2[IDAS66]",
    "IDAS2[IDAS70]",
    "IDAS2[IDAS76]",
    "IDAS2[IDAS82]",
    "IDAS2[IDAS86]",
]

# 10) Euphorie: 71, 77, 87, 91, 96
euphoria = [
    "IDAS2[IDAS71]",
    "IDAS2[IDAS77]",
    "IDAS2[IDAS87]",
    "IDAS2[IDAS91]",
    "IDAS2[IDAS96]",
]

# 11) Soziale Angst: 15, 18, 20, 40, 46, 98
social_anx = [
    "IDAS1[IDAS15]",
    "IDAS1[IDAS18]",
    "IDAS1[IDAS20]",
    "IDAS1[IDAS40]",
    "IDAS2[IDAS46]",
    "IDAS2[IDAS98]",
]

# 12) Klaustrophobie: 73, 79, 83, 89, 93
claustrophobia = [
    "IDAS2[IDAS73]",
    "IDAS2[IDAS79]",
    "IDAS2[IDAS83]",
    "IDAS2[IDAS89]",
    "IDAS2[IDAS93]",
]

# 13) Traumatische Intrusionen: 14, 28, 34, 41
traumatic_intrusions = [
    "IDAS1[IDAS14]",
    "IDAS1[IDAS28]",
    "IDAS1[IDAS34]",
    "IDAS1[IDAS41]",
]

# 14) Traumatische Vermeidung: 72, 78, 88, 92
traumatic_avoidance = [
    "IDAS2[IDAS72]",
    "IDAS2[IDAS78]",
    "IDAS2[IDAS88]",
    "IDAS2[IDAS92]",
]

# 15) Ordnungszwang: 64, 68, 81, 84, 94
compulsion_order = [
    "IDAS2[IDAS64]",
    "IDAS2[IDAS68]",
    "IDAS2[IDAS81]",
    "IDAS2[IDAS84]",
    "IDAS2[IDAS94]",
]

# 16) Reinigungszwang: 65, 69, 75, 85, 90, 95, 97
compulsion_clean = [
    "IDAS2[IDAS65]",
    "IDAS2[IDAS69]",
    "IDAS2[IDAS75]",
    "IDAS2[IDAS85]",
    "IDAS2[IDAS90]",
    "IDAS2[IDAS95]",
    "IDAS2[IDAS97]",
]

# 17) Kontrollzwang: 67, 74, 80
compulsion_control = ["IDAS2[IDAS67]", "IDAS2[IDAS74]", "IDAS2[IDAS80]"]

# 18) Panik: 7, 16, 32, 38, 44, 48, 55, 57
panic = [
    "IDAS1[IDAS07]",
    "IDAS1[IDAS16]",
    "IDAS1[IDAS32]",
    "IDAS1[IDAS38]",
    "IDAS1[IDAS44]",
    "IDAS2[IDAS48]",
    "IDAS2[IDAS55]",
    "IDAS2[IDAS57]",
]

# Initialize dimension lists
dysphoria_vals = list()
fatigue_vals = list()
insomnia_vals = list()
suicidality_vals = list()
incr_appetite_vals = list()
loss_appetite_vals = list()
wellbeing_vals = list()
moodiness_vals = list()
mania_vals = list()
euphoria_vals = list()
social_anx_vals = list()
claustrophobia_vals = list()
traumatic_intrusions_vals = list()
traumatic_avoidance_vals = list()
compulsion_order_vals = list()
compulsion_clean_vals = list()
compulsion_control_vals = list()
panic_vals = list()

# Cycle over subjects
qname = "IDAS"
for subjInd, subj in enumerate(subjects):

    # Re-initialize sum score variables and counter
    dysphoria_sum = 0
    fatigue_sum = 0
    insomnia_sum = 0
    suicidality_sum = 0
    incr_appetite_sum = 0
    loss_appetite_sum = 0
    wellbeing_sum = 0
    moodiness_sum = 0
    mania_sum = 0
    euphoria_sum = 0
    social_anx_sum = 0
    claustrophobia_sum = 0
    traumatic_intrusions_sum = 0
    traumatic_avoidance_sum = 0
    compulsion_order_sum = 0
    compulsion_clean_sum = 0
    compulsion_control_sum = 0
    panic_sum = 0
    counter = 0

    # Select all questionnaire items of current subject
    df2 = qns_data[qns_data["subj_num"] == subj].filter(regex=rf"^{qname}.*")

    # Cycle over each question for current participant
    for col in df2.columns:

        # Only consider non-NaNs and update respective variables
        if not np.isnan(df2[col].iloc[0]):
            if col in dysphoria:
                dysphoria_sum += df2[col].iloc[0]
            elif col in fatigue:
                fatigue_sum += df2[col].iloc[0]
            elif col in insomnia:
                insomnia_sum += df2[col].iloc[0]
            elif col in suicidality:
                suicidality_sum += df2[col].iloc[0]
            elif col in incr_appetite:
                incr_appetite_sum += df2[col].iloc[0]
            elif col in loss_appetite:
                loss_appetite_sum += df2[col].iloc[0]
            elif col in wellbeing:
                wellbeing_sum += df2[col].iloc[0]
            elif col in moodiness:
                moodiness_sum += df2[col].iloc[0]
            elif col in mania:
                mania_sum += df2[col].iloc[0]
            elif col in euphoria:
                euphoria_sum += df2[col].iloc[0]
            elif col in social_anx:
                social_anx_sum += df2[col].iloc[0]
            elif col in claustrophobia:
                claustrophobia_sum += df2[col].iloc[0]
            elif col in traumatic_intrusions:
                traumatic_intrusions_sum += df2[col].iloc[0]
            elif col in traumatic_avoidance:
                traumatic_avoidance_sum += df2[col].iloc[0]
            elif col in compulsion_order:
                compulsion_order_sum += df2[col].iloc[0]
            elif col in compulsion_clean:
                compulsion_clean_sum += df2[col].iloc[0]
            elif col in compulsion_control:
                compulsion_control_sum += df2[col].iloc[0]
            elif col in panic:
                panic_sum += df2[col].iloc[0]
            else:
                sys.exit("delta_t is NaN")

        # Update counter
        counter += 1

    # Compute total number of included questions
    total_length = (
        len(dysphoria)
        + len(fatigue)
        + len(insomnia)
        + len(suicidality)
        + len(incr_appetite)
        + len(loss_appetite)
        + len(wellbeing)
        + len(moodiness)
        + len(mania)
        + len(euphoria)
        + len(social_anx)
        + len(claustrophobia)
        + len(traumatic_intrusions)
        + len(traumatic_avoidance)
        + len(compulsion_order)
        + len(compulsion_clean)
        + len(compulsion_control)
        + len(panic)
    )

    # Compare counter and total N of questions included
    if counter != total_length:
        message = "counter = " + str(counter) + " but N items = " + str(total_length)
        print(message)
        sys.exit(1)
    else:
        # Add sum scores to lists
        dysphoria_vals.append(dysphoria_sum)
        fatigue_vals.append(fatigue_sum)
        insomnia_vals.append(insomnia_sum)
        suicidality_vals.append(suicidality_sum)
        incr_appetite_vals.append(incr_appetite_sum)
        loss_appetite_vals.append(loss_appetite_sum)
        wellbeing_vals.append(wellbeing_sum)
        moodiness_vals.append(moodiness_sum)
        mania_vals.append(mania_sum)
        euphoria_vals.append(euphoria_sum)
        social_anx_vals.append(social_anx_sum)
        claustrophobia_vals.append(claustrophobia_sum)
        traumatic_intrusions_vals.append(traumatic_intrusions_sum)
        traumatic_avoidance_vals.append(traumatic_avoidance_sum)
        compulsion_order_vals.append(compulsion_order_sum)
        compulsion_clean_vals.append(compulsion_clean_sum)
        compulsion_control_vals.append(compulsion_control_sum)
        panic_vals.append(panic_sum)

# Add dimension sum scores
df_totalscore["dysphoria"] = dysphoria_vals
df_totalscore["fatigue"] = fatigue_vals
df_totalscore["insomnia"] = insomnia_vals
df_totalscore["suicidality"] = suicidality_vals
df_totalscore["incr_appetite"] = incr_appetite_vals
df_totalscore["loss_appetite"] = loss_appetite_vals
df_totalscore["wellbeing"] = wellbeing_vals
df_totalscore["moodiness"] = moodiness_vals
df_totalscore["mania"] = mania_vals
df_totalscore["euphoria"] = euphoria_vals
df_totalscore["social_anx"] = social_anx_vals
df_totalscore["claustrophobia"] = claustrophobia_vals
df_totalscore["traumatic_intrusions"] = traumatic_intrusions_vals
df_totalscore["traumatic_avoidance"] = traumatic_avoidance_vals
df_totalscore["compulsion_order"] = compulsion_order_vals
df_totalscore["compulsion_clean"] = compulsion_clean_vals
df_totalscore["compulsion_control"] = compulsion_control_vals
df_totalscore["panic"] = panic_vals

# Save data frame
df_totalscore.name = "questionnaires_totalscores"
safe_save_dataframe(df_totalscore)
