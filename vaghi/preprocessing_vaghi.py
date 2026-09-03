if __name__ == "__main__":

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

    import json
    import os

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy
    import seaborn as sns
    from allinpy import latex_plt
    from rbmpy.utilities import circ_dist

    from FOR_1_Paper.for_utilities import safe_save_dataframe

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    # Turn interactive mode on
    plt.ion()

    # Get home directory
    paths = os.getcwd()
    path = paths.split(os.path.sep)
    home_dir = path[1]

    # -----------------------------------------
    # Translate Vaghi data into FOR data format
    # -----------------------------------------

    # Load participant data
    with open("for_data/vaghi_data/data.json") as f:
        data = json.load(f)

    # Extract variables
    sub = data["sub"]  # participant ID
    trial = data["trial"]  # trial number
    block = data["block"]  # block number
    bucketPosition = data["bucketPosition"]  # participant prediction
    hit = data["hit"]  # hit indicator
    bucketRT = data["bucketRT"]  # response time
    group = data["group"]  # group indicator

    # We don't need the following variables:
    # confidence = data["confidence"]
    # bet = data["bet"]
    # betRT = data["betRT"]
    # trialReward = data["trialReward"]
    # blockReward = data["blockReward"]
    # totReward = data["totReward"]

    # Load stimulus data
    with open("for_data/vaghi_data/stim.json") as f:
        stim = json.load(f)

    # Extract variables
    samples = stim["samples"]  # outcome x_t
    refLocation = stim["refLocation"]  # mean mu_t
    std = stim["std"]  # noise level sigma_t

    # Manually compute new-block indicator
    new_block = np.full(len(block), np.nan)
    new_block[0] = True
    new_block[1:] = (np.array(block)[:-1] - np.array(block)[1:]) != 0

    # Recode the last entry in variables of interest of each block to nan
    # to avoid that data of different participants are mixed
    to_nan = np.zeros(len(new_block))
    to_nan[:-1] = new_block[1:]
    to_nan[-1] = 1  # manually added, because no new block after the last trial

    # Combine everything in one data frame
    # ------------------------------------
    vaghi_data = pd.DataFrame()
    vaghi_data["ID"] = sub  # original participant ID
    vaghi_data["subj_num"] = pd.factorize(vaghi_data["ID"])[0] + 1  # subject number
    vaghi_data["trial"] = trial  # trial number
    vaghi_data["block"] = block  # block number
    vaghi_data["new_block"] = new_block  # new-block indicator
    vaghi_data["b_t_rad"] = np.deg2rad(bucketPosition)  # bucket position in radians
    vaghi_data["x_t_rad"] = np.deg2rad(samples)  # outcome position in radians
    vaghi_data["mu_t_rad"] = np.deg2rad(refLocation)  # true mean position in radians
    vaghi_data["e_t_rad"] = circ_dist(
        vaghi_data["mu_t_rad"], vaghi_data["b_t_rad"]
    )  # estimation error in radians
    vaghi_data["sigma"] = np.deg2rad(std)  # standard deviation of outcomes in radians
    vaghi_data["hit"] = hit  # hit indicator
    vaghi_data["hit_dummy"] = np.nan  # hit indicator dummy variable
    vaghi_data.loc[vaghi_data["hit"] == 1, "hit_dummy"] = 1
    vaghi_data.loc[vaghi_data["hit"] == 0, "hit_dummy"] = -1
    vaghi_data["kappa_dummy"] = 0  # since only 1 noise condition, dummy is always 0
    vaghi_data["delta_t_rad"] = circ_dist(
        vaghi_data["x_t_rad"], vaghi_data["b_t_rad"]
    )  # prediction error in radians
    vaghi_data.loc[to_nan == 1, "delta_t_rad"] = np.nan
    vaghi_data["a_t_rad"] = np.nan  # initialize update
    vaghi_data.loc[vaghi_data.index[:-1], "a_t_rad"] = circ_dist(
        vaghi_data.loc[vaghi_data.index[1:], "b_t_rad"],
        vaghi_data.loc[vaghi_data.index[:-1], "b_t_rad"],
    )  # compute update in radians
    vaghi_data.loc[to_nan == 1, "a_t_rad"] = np.nan
    vaghi_data["a_t"] = np.rad2deg(vaghi_data["a_t_rad"])
    vaghi_data["pers"] = (
        abs(vaghi_data["a_t"]) <= 1.0e-1
    )  # compute perseveration trials
    vaghi_data["group"] = group  # group
    vaghi_data["v_t"] = False  # no catch trials
    vaghi_data["v_dummy"] = 0  # because no catch trials, dummy is always 0

    # Save data
    vaghi_data.name = "vaghi_data_prepr"
    safe_save_dataframe(vaghi_data)

    # ---------------------------
    # Run some descriptive checks
    # ---------------------------

    # 1. Check that perseveration is computed correctly
    # -------------------------------------------------

    df_pers = pd.DataFrame()
    df_pers["subj_num"] = vaghi_data["subj_num"].copy()
    df_pers["pers_e_5"] = (
        abs(vaghi_data["a_t"]) <= 1.0e-5
    )  # compute perseveration trials
    df_pers["pers_e_1"] = (
        abs(vaghi_data["a_t"]) <= 1.0e-1
    )  # compute perseveration trials
    df_pers["pers_0"] = abs(vaghi_data["a_t"]) == 0  # compute perseveration trials

    mean_e_5 = df_pers.groupby("subj_num")["pers_e_5"].mean()
    mean_e_1 = df_pers.groupby("subj_num")["pers_e_1"].mean()
    mean_0 = df_pers.groupby("subj_num")["pers_0"].mean()

    # Plot histogram for comparison
    plt.figure()
    plt.hist(mean_e_5, label="e-5", alpha=0.5)
    plt.hist(mean_e_1, label="e-1", alpha=0.5)
    plt.hist(mean_0, label="0", alpha=0.5)
    plt.legend()
    plt.title("Different perseveration criteria")
    plt.xlabel("Perseveration probability")
    sns.despine()

    # Print mean perseveration probability for each criterion
    print("Comparing different perseveration criteria:")
    print("e-5:" + str(np.mean(mean_e_5)))
    print("e-1:" + str(np.mean(mean_e_1)))
    print("0:" + str(np.mean(mean_0)))

    # 2. Compare perseveration between groups
    # ---------------------------------------

    subject_groups = vaghi_data.groupby("subj_num")["group"].first()
    n_subj_g1 = sum(subject_groups == 1)
    n_subj_g0 = sum(subject_groups == 0)

    # Compute perseveration probability
    pers_prob_g1 = np.mean(vaghi_data[vaghi_data["group"] == 1]["pers"])
    pers_prob_g0 = np.mean(vaghi_data[vaghi_data["group"] == 0]["pers"])
    pers_prob_g0_sem = np.std(vaghi_data[vaghi_data["group"] == 0]["pers"]) / np.sqrt(
        n_subj_g0
    )
    pers_prob_g1_sem = np.std(vaghi_data[vaghi_data["group"] == 1]["pers"]) / np.sqrt(
        n_subj_g1
    )

    # Compute mean of "pers" for each ID
    pers_mean_by_id_group = (
        vaghi_data.groupby(["ID", "group"])["pers"].mean().reset_index()
    )

    # Run t-test
    res = scipy.stats.ttest_ind(
        pers_mean_by_id_group.loc[pers_mean_by_id_group["group"] == 0, "pers"],
        pers_mean_by_id_group.loc[pers_mean_by_id_group["group"] == 1, "pers"],
    )

    # Plot perseveration probability for each group
    plt.figure()
    plt.bar(
        [0, 1],
        [pers_prob_g0, pers_prob_g1],
        yerr=[pers_prob_g0_sem, pers_prob_g1_sem],
    )
    plt.xticks([0, 1], ["OCD", "Control"])
    plt.ylabel("Perseveration probability")
    sns.despine()
    plt.title(r"$p$ = " + str(res.pvalue.round(3)))
    save_name = (
        "/"
        + home_dir
        + "/rasmus/Dropbox/for_analyses/FOR_1_Paper/figures/vaghi_pers.png"
    )
    plt.savefig(save_name, dpi=400)

    # 3. Check single-trial learning rate
    # -----------------------------------

    LR = vaghi_data["a_t_rad"] / vaghi_data["delta_t_rad"]
    LR[np.isinf(LR)] = np.nan
    LR[LR > 3] = 3
    LR[LR < -1] = -1
    plt.figure()
    plt.hist(LR, bins=100)
    plt.xlabel("Learning rate")
    sns.despine()

    # 4. Typical regression plot per subject
    # --------------------------------------

    single_subject_plot = False
    if single_subject_plot:

        # Cycle over subjects
        IDs = np.unique(vaghi_data["ID"])
        for which_id in range(len(IDs)):

            plt.figure()
            ax = plt.gca()
            ax.axhline(0, color="gray", linestyle="-")
            ax.axvline(0, color="gray", linestyle="-")
            ax.axline(
                (0, 0),
                slope=1,
                color="gray",
            )
            plt.plot(
                vaghi_data.loc[vaghi_data["ID"] == IDs[which_id], "delta_t_rad"],
                vaghi_data.loc[vaghi_data["ID"] == IDs[which_id], "a_t_rad"],
                ".",
            )
            plt.xlabel("Prediction error")
            plt.ylabel("Update")
            sns.despine()
            plt.savefig("figures/vaghi/reg_plot/" + str(which_id) + ".png", dpi=400)
            plt.close()

    # 5. Typical regression plot across all subjects
    # ----------------------------------------------

    plt.figure()
    ax = plt.gca()
    ax.axhline(0, color="gray", linestyle="-")
    ax.axvline(0, color="gray", linestyle="-")
    ax.axline(
        (0, 0),
        slope=1,
        color="gray",
    )
    plt.plot(vaghi_data["delta_t_rad"], vaghi_data["a_t_rad"], ".")
    plt.xlabel("Prediction error")
    plt.ylabel("Update")
    sns.despine()
    plt.ioff()
    plt.show()
