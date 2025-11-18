"""Simple plot for model comparison based on BIC."""

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from allinpy import latex_plt

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Use preferred backend for Linux, or just take default
try:
    matplotlib.use("Qt5Agg")
except ImportError:
    pass

model_1_1 = pd.read_pickle("for_data/regression_model_1_1_50_sp.pkl")
model_1_2 = pd.read_pickle("for_data/regression_model_1_2_50_sp.pkl")
model_1_3 = pd.read_pickle("for_data/regression_model_1_3_50_sp.pkl")
model_1_4 = pd.read_pickle("for_data/regression_model_1_4_50_sp.pkl")
model_1_5 = pd.read_pickle("for_data/regression_model_1_5_50_sp.pkl")
model_1_6 = pd.read_pickle("for_data/regression_model_1_6_50_sp.pkl")

model_2_1 = pd.read_pickle("for_data/regression_model_2_1_50_sp.pkl")
model_2_2 = pd.read_pickle("for_data/regression_model_2_2_50_sp.pkl")
model_2_3 = pd.read_pickle("for_data/regression_model_2_3_50_sp.pkl")
model_2_4 = pd.read_pickle("for_data/regression_model_2_4_50_sp.pkl")

plt.figure()
bic_values = [
    sum(model_1_1["BIC"]),
    sum(model_1_2["BIC"]),
    sum(model_1_3["BIC"]),
    sum(model_1_4["BIC"]),
    sum(model_1_5["BIC"]),
    sum(model_1_6["BIC"]),
    sum(model_2_1["BIC"]),
    sum(model_2_2["BIC"]),
    sum(model_2_3["BIC"]),
    sum(model_2_4["BIC"]),
]
plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], bic_values)
plt.xticks(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [
        "Model 1.1",
        "Model 1.2",
        "Model 1.3",
        "Model 1.4",
        "Model 1.5",
        "Model 1.6",
        "Model 2.1",
        "Model 2.2",
        "Model 2.3",
        "Model 2.4",
    ],
)
plt.ylabel("Sum BIC")
sns.despine()

print(
    "Best model: ",
    bic_values.index(max(bic_values)) + 1,
    "with BIC = ",
    max(bic_values),
)  # +1 because starts from 0


plt.show()