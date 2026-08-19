"""Simple plot for model comparison based on BIC."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from allinpy import latex_plt

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)


def regression_comparison(
    regression_1,
    regression_2,
    regression_3,
    regression_4,
    regression_5,
    regression_6,
    regression_7,
    regression_8,
    regression_9,
    regression_10,
    regression_11,
    regression_12,
    regression_13,
    regression_14,
    regression_15,
    regression_16,
):
    """
    Compare and visualize Bayesian Information Criterion (BIC) values across multiple regression models.

    This function computes the sum of BIC values for multiple regression models, visualizes them
    in a bar chart, and identifies the model with the highest BIC value.

    Parameters
    ----------
    regression_1 : dict
        Dictionary containing the BIC values for the first model under the key "BIC".
    regression_2 : dict
        Dictionary containing the BIC values for the second model under the key "BIC".
    regression_3 : dict
        Dictionary containing the BIC values for the third model under the key "BIC".
    regression_4 : dict
        Dictionary containing the BIC values for the fourth model under the key "BIC".
    regression_5 : dict
        Dictionary containing the BIC values for the fifth model under the key "BIC".
    regression_6 : dict
        Dictionary containing the BIC values for the sixth model under the key "BIC".
    regression_7 : dict
        Dictionary containing the BIC values for the seventh model under the key "BIC".
    regression_8 : dict
        Dictionary containing the BIC values for the eighth model under the key "BIC".
    regression_9 : dict
        Dictionary containing the BIC values for the ninth model under the key "BIC".
    regression_10 : dict
        Dictionary containing the BIC values for the tenth model under the key "BIC".
    regression_11 : dict
        Dictionary containing the BIC values for the eleventh model under the key "BIC".
    regression_12 : dict
        Dictionary containing the BIC values for the twelfth model under the key "BIC".
    regression_13 : dict
        Dictionary containing the BIC values for the thirteenth model under the key "BIC".
    regression_14 : dict
        Dictionary containing the BIC values for the fourteenth model under the key "BIC".
    regression_15 : dict
        Dictionary containing the BIC values for the fifteenth model under the key "BIC".
    regression_16 : dict
        Dictionary containing the BIC values for the sixteenth model under the key "BIC".

    Returns
    -------
    None
    """

    plt.figure(figsize=(10, 5))
    bic_values = [
        sum(regression_1["BIC"]),
        sum(regression_2["BIC"]),
        sum(regression_3["BIC"]),
        sum(regression_4["BIC"]),
        sum(regression_5["BIC"]),
        sum(regression_6["BIC"]),
        sum(regression_7["BIC"]),
        sum(regression_8["BIC"]),
        sum(regression_9["BIC"]),
        sum(regression_10["BIC"]),
        sum(regression_11["BIC"]),
        sum(regression_12["BIC"]),
        sum(regression_13["BIC"]),
        sum(regression_14["BIC"]),
        sum(regression_15["BIC"]),
        sum(regression_16["BIC"]),
    ]
    plt.bar(np.arange(len(bic_values)), bic_values)

    model_names = [
        "Model 11",
        "Model 12",
        "Model 13",
        "Model 14",
        "Model 15",
        "Model 16",
        "Model 17",
        "Model 18",
        "Model 21",
        "Model 22",
        "Model 23",
        "Model 24",
        "Model 25",
        "Model 26",
        "Model 27",
        "Model 28",
    ]

    plt.xticks(
        np.arange(len(bic_values)),
        model_names,
    )
    plt.ylabel("Sum BIC")
    sns.despine()

    print(
        "Best model: ",
        model_names[bic_values.index(max(bic_values))],
        "with BIC = ",
        max(bic_values),
    )


model_11 = pd.read_pickle(
    "for_data/sca_for/regression_11_50sp_3e58055ad76320666db19165f61d0815.pkl"
)
model_12 = pd.read_pickle(
    "for_data/sca_for/regression_12_50sp_005def48343266e3c8483043f8883cf5.pkl"
)
model_13 = pd.read_pickle(
    "for_data/sca_for/regression_13_50sp_f7a023ce7b8999b8e48c8b7988fba9c9.pkl"
)
model_14 = pd.read_pickle(
    "for_data/sca_for/regression_14_50sp_5c630de6448b48802d8084d7b9610fcf.pkl"
)
model_15 = pd.read_pickle(
    "for_data/sca_for/regression_15_50sp_134dcd524d41e2b7d0e773072a692f5d.pkl"
)
model_16 = pd.read_pickle(
    "for_data/sca_for/regression_16_50sp_f188bb733c0bcf04f452189238259449.pkl"
)
model_17 = pd.read_pickle(
    "for_data/sca_for/regression_17_50sp_b64001f2d4407c51291512eac07c9015.pkl"
)
model_18 = pd.read_pickle(
    "for_data/sca_for/regression_18_50sp_0166b7e350694116241d70cd1cb4eb36.pkl"
)
model_21 = pd.read_pickle(
    "for_data/sca_for/regression_21_50sp_8465a564cb835ac360895bb977282d55.pkl"
)
model_22 = pd.read_pickle(
    "for_data/sca_for/regression_22_50sp_aa8b1d4ea8edb9a58ed0980db68caffd.pkl"
)
model_23 = pd.read_pickle(
    "for_data/sca_for/regression_23_50sp_781f774709e886f34165b4f16a12fc8c.pkl"
)
model_24 = pd.read_pickle(
    "for_data/sca_for/regression_24_50sp_f32900a2b5cdf4a0938f58e94e892d5e.pkl"
)
model_25 = pd.read_pickle(
    "for_data/sca_for/regression_25_50sp_d3676ab422fd64fa166aedec67a81b2d.pkl"
)
model_26 = pd.read_pickle(
    "for_data/sca_for/regression_26_50sp_66ce6627d7173840291de59d700e09bf.pkl"
)
model_27 = pd.read_pickle(
    "for_data/sca_for/regression_27_50sp_14d7708aba0c45b68817a3d3030cda81.pkl"
)
model_28 = pd.read_pickle(
    "for_data/sca_for/regression_28_50sp_01f61b6b58bebed02cadeaab5bceba50.pkl"
)


regression_comparison(
    model_11,
    model_12,
    model_13,
    model_14,
    model_15,
    model_16,
    model_17,
    model_18,
    model_21,
    model_22,
    model_23,
    model_24,
    model_25,
    model_26,
    model_27,
    model_28,
)


model_11 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_11_50sp_3e58055ad76320666db19165f61d0815.pkl"
)
model_12 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_12_50sp_005def48343266e3c8483043f8883cf5.pkl"
)
model_13 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_13_50sp_f7a023ce7b8999b8e48c8b7988fba9c9.pkl"
)
model_14 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_14_50sp_5c630de6448b48802d8084d7b9610fcf.pkl"
)
model_15 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_15_50sp_134dcd524d41e2b7d0e773072a692f5d.pkl"
)
model_16 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_16_50sp_f188bb733c0bcf04f452189238259449.pkl"
)
model_17 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_17_50sp_b64001f2d4407c51291512eac07c9015.pkl"
)
model_18 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_18_50sp_0166b7e350694116241d70cd1cb4eb36.pkl"
)
model_21 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_21_50sp_8465a564cb835ac360895bb977282d55.pkl"
)
model_22 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_22_50sp_aa8b1d4ea8edb9a58ed0980db68caffd.pkl"
)
model_23 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_23_50sp_781f774709e886f34165b4f16a12fc8c.pkl"
)
model_24 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_24_50sp_f32900a2b5cdf4a0938f58e94e892d5e.pkl"
)
model_25 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_25_50sp_d3676ab422fd64fa166aedec67a81b2d.pkl"
)
model_26 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_26_50sp_66ce6627d7173840291de59d700e09bf.pkl"
)
model_27 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_27_50sp_14d7708aba0c45b68817a3d3030cda81.pkl"
)
model_28 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/regression_28_50sp_01f61b6b58bebed02cadeaab5bceba50.pkl"
)

regression_comparison(
    model_11,
    model_12,
    model_13,
    model_14,
    model_15,
    model_16,
    model_17,
    model_18,
    model_21,
    model_22,
    model_23,
    model_24,
    model_25,
    model_26,
    model_27,
    model_28,
)

plt.show()
