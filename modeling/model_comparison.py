"""Simple RBM model comparison."""

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


def rbm_comparison(
    rbm_1: pd.DataFrame,
    rbm_2: pd.DataFrame,
    rbm_3: pd.DataFrame,
    rbm_4: pd.DataFrame,
    rbm_5: pd.DataFrame,
    rbm_6: pd.DataFrame,
    rbm_7: pd.DataFrame,
    rbm_8: pd.DataFrame,
):
    """Visualizes and compares the Bayesian Information Criterion (BIC) scores of multiple
    models and provides detailed histograms and statistical analysis for the parameters
    of a specific model (Model 18). The function identifies the best model based on the
    BIC score and generates informative plots to evaluate key parameter distributions.

    Parameters
    ----------
    rbm_1 : pd.DataFrame
        Dictionary containing data for Model 1. Must include a key `"BIC"` with a
        list of BIC values.
    rbm_2 : pd.DataFrame
        Dictionary containing data for Model 2. Must include a key `"BIC"` with a
        list of BIC values.
    rbm_3 : pd.DataFrame
        Dictionary containing data for Model 3. Must include a key `"BIC"` with a
        list of BIC values.
    rbm_4 : pd.DataFrame
        Dictionary containing data for Model 4. Must include a key `"BIC"` with a
        list of BIC values.
    rbm_5 : pd.DataFrame
        Dictionary containing data for Model 5. Must include a key `"BIC"` with a
        list of BIC values.
    rbm_6 : pd.DataFrame
        Dictionary containing data for Model 6. Must include a key `"BIC"` with a
        list of BIC values.
    rbm_7 : pd.DataFrame
        Dictionary containing data for Model 7. Must include a key `"BIC"` with a
        list of BIC values.
    rbm_8 : pd.DataFrame
        Dictionary containing data for Model 8. Must include keys `"BIC"`, `"omikron_0"`,
        `"omikron_1"`, `"s"`, and `"u"`. Each key should have a corresponding list or array
        of values for analysis.

    Returns
    -------
    None
        This function does not return any value.
    """

    plt.figure()
    bic_values = [
        sum(rbm_1["BIC"]),
        sum(rbm_2["BIC"]),
        sum(rbm_3["BIC"]),
        sum(rbm_4["BIC"]),
        sum(rbm_5["BIC"]),
        sum(rbm_6["BIC"]),
        sum(rbm_7["BIC"]),
        sum(rbm_8["BIC"]),
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

    plt.figure()
    plt.subplot(2, 3, 1)
    omikron_0 = rbm_8["omikron_0"]
    plt.hist(omikron_0, bins=20)
    plt.title(
        "Motor Noise: mean = "
        + str(np.round(np.mean(omikron_0), decimals=2))
        + ", SD = "
        + str(np.round(np.std(omikron_0), decimals=2))
    )

    plt.subplot(2, 3, 2)
    omikron_1 = rbm_8["omikron_1"]
    plt.hist(omikron_1, bins=20)
    plt.title(
        "Learning-Rate Noise: mean = "
        + str(np.round(np.mean(omikron_1), decimals=2))
        + ", SD = "
        + str(np.round(np.std(omikron_1), decimals=2))
    )

    plt.subplot(2, 3, 3)
    s = rbm_8["s"]
    plt.hist(s, bins=20, label=["s"])
    plt.title(
        "Surprise Sensitivity: mean = "
        + str(np.round(np.mean(s), decimals=2))
        + ", SD = "
        + str(np.round(np.std(s), decimals=2))
    )

    plt.subplot(2, 3, 4)
    u = rbm_8["u"]
    plt.hist(u, bins=20, label=["u"])
    plt.title(
        "Uncertainty Underestimation: mean = "
        + str(np.round(np.mean(u), decimals=2))
        + ", SD = "
        + str(np.round(np.std(u), decimals=2))
    )

    plt.subplot(2, 3, 5)
    sigma_H = rbm_8["sigma_H"]
    plt.hist(sigma_H, bins=20, label=["u"])
    plt.title(
        "Catch trial: mean = "
        + str(np.round(np.mean(sigma_H), decimals=2))
        + ", SD = "
        + str(np.round(np.std(sigma_H), decimals=2))
    )

    plt.tight_layout()
    sns.despine()


# ----------------------
# 1. Empirical estimates
# ----------------------

rbm_11 = pd.read_pickle(
    "for_data/sca_for/rbm_11_10sp_184a539ed3ea14a6801b14991444dec9.pkl"
)
rbm_12 = pd.read_pickle(
    "for_data/sca_for/rbm_12_10sp_0a55d1b66d149f8408aa63949dc11927.pkl"
)
rbm_13 = pd.read_pickle(
    "for_data/sca_for/rbm_13_10sp_567c32f3bd2b58cb109555654f35c1cb.pkl"
)
rbm_14 = pd.read_pickle(
    "for_data/sca_for/rbm_14_10sp_ec993b640c4c5ff3ce4473e811cdab80.pkl"
)
rbm_15 = pd.read_pickle(
    "for_data/sca_for/rbm_15_10sp_793d77840a4381b3fb41532ab1319111.pkl"
)
rbm_16 = pd.read_pickle(
    "for_data/sca_for/rbm_16_10sp_a50e2be97241085f5a0444fa11b5e007.pkl"
)
rbm_17 = pd.read_pickle(
    "for_data/sca_for/rbm_17_10sp_d898c2b48e6f41bc63b0bf99bae23d61.pkl"
)
rbm_18 = pd.read_pickle(
    "for_data/sca_for/rbm_18_10sp_4d6aaf48373d27c6a5b68c6eb32ee83d.pkl"
)

rbm_comparison(rbm_11, rbm_12, rbm_13, rbm_14, rbm_15, rbm_16, rbm_17, rbm_18)

# ----------------------
# 2. Simulated estimates
# ----------------------

rbm_11 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/rbm_11_10sp_184a539ed3ea14a6801b14991444dec9.pkl"
)
rbm_12 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/rbm_12_10sp_0a55d1b66d149f8408aa63949dc11927.pkl"
)
rbm_13 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/rbm_13_10sp_567c32f3bd2b58cb109555654f35c1cb.pkl"
)
rbm_14 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/rbm_14_10sp_ec993b640c4c5ff3ce4473e811cdab80.pkl"
)
rbm_15 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/rbm_15_10sp_793d77840a4381b3fb41532ab1319111.pkl"
)
rbm_16 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/rbm_16_10sp_a50e2be97241085f5a0444fa11b5e007.pkl"
)
rbm_17 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/rbm_17_10sp_d898c2b48e6f41bc63b0bf99bae23d61.pkl"
)
rbm_18 = pd.read_pickle(
    "for_data/sca_task_agent_N200_T400_seed123/rbm_18_10sp_4d6aaf48373d27c6a5b68c6eb32ee83d.pkl"
)


rbm_comparison(rbm_11, rbm_12, rbm_13, rbm_14, rbm_15, rbm_16, rbm_17, rbm_18)

plt.show()
