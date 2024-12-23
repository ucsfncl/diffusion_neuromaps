# We do not include the metadata in the repository.
# The metadata is available from the Human Connectome Project once you've filled out the data access agreement.
# Restricted data requires an additional application process.
# We do provide the full CV results in data/ridge/r2_metric.csv and data/ridge/r2_factor.csv.
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, RidgeCV
from sklearn.model_selection import cross_val_score, RepeatedKFold
import scipy
import scipy.stats


# Load the data
data_path = "diffusion_neuromaps/data"
subjects = np.loadtxt("diffusion_neuromaps/data/hcp_qc_subjects.txt", dtype=str)
subjects = np.array(subjects)
hcp_restricted = pd.read_csv("dsn/data/hcp_restricted.csv")
hcp_metadata = pd.read_csv("HCP_METADATA.csv")
targets = ["CogFluidComp_Unadj", "CogEarlyComp_Unadj", "CogTotalComp_Unadj", "CogCrystalComp_Unadj"]
restricted_targets = ["Age_in_Yrs"]
hcp_metadata = hcp_metadata[hcp_metadata["Subject"].isin(subjects.astype(int))]
hcp_restricted = hcp_restricted[hcp_restricted["Subject"].isin(subjects.astype(int))]

metrics = ["fa", "md", "ad", "rd", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
           "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]
factors = ["F1", "F2", "F3", "F4"]
r2_df = pd.DataFrame(columns=["type", "r2", "target"])
n_repeats = 20
cv = RepeatedKFold(n_repeats=n_repeats, n_splits=5, random_state=42)

for metric in metrics:
    df = pd.read_csv(os.path.join(data_path, "glasser", "subject", f"{metric}_glasser.csv"), index_col=0, header=0)
    df = (df - df.mean(axis=0)) / df.std(axis=0)
    df = df.values
    
    for target in targets:
        y = hcp_metadata[target].values
        X = df[~np.isnan(y)]
        y = y[~np.isnan(y)]
        print(f"{metric}: {target}")
        ridge = RidgeCV(alphas=np.logspace(-6, 6, 20), scoring="r2")
        scores = cross_val_score(ridge, X, y, cv=cv, scoring="r2")
        print(np.mean(scores), np.std(scores))
        r2_df = pd.concat([r2_df, pd.DataFrame({"type": [metric] * len(scores), "r2": scores, "target": [target] * len(scores)})])
    
    for target in restricted_targets:
        y = hcp_restricted[target].values
        X = df[~np.isnan(y)]
        y = y[~np.isnan(y)]
        print(f"{metric}: {target}")
        ridge = RidgeCV(alphas=np.logspace(-6, 6, 20), scoring="r2")
        scores = cross_val_score(ridge, X, y, cv=cv, scoring="r2")
        print(np.mean(scores), np.std(scores))
        r2_df = pd.concat([r2_df, pd.DataFrame({"type": [metric] * len(scores), "r2": scores, "target": [target] * len(scores)})])

r2_df["r2"] = np.clip(r2_df["r2"], 0, 1)
r2_df.to_csv("diffusion_neuromaps/data/ridge/r2_metric.csv", index=False)

for factor in factors:
    df = pd.read_csv(os.path.join(data_path, "glasser", "ave", f"{factor}_glasser_ave.csv"), index_col=0, header=0)
    df = (df - df.mean(axis=0)) / df.std(axis=0)
    df = df.values

    for target in targets:
        y = hcp_metadata[target].values
        X = df[~np.isnan(y)]
        y = y[~np.isnan(y)]
        print(f"{factor}: {target}")
        ridge = RidgeCV(alphas=np.logspace(-6, 6, 20), scoring="r2")
        scores = cross_val_score(ridge, X, y, cv=cv, scoring="r2")
        print(np.mean(scores), np.std(scores))
        r2_df = pd.concat([r2_df, pd.DataFrame({"type": [factor] * len(scores), "r2": scores, "target": [target] * len(scores)})])

    for target in restricted_targets:
        y = hcp_restricted[target].values
        X = df[~np.isnan(y)]
        y = y[~np.isnan(y)]
        print(f"{factor}: {target}")
        ridge = RidgeCV(alphas=np.logspace(-6, 6, 20), scoring="r2")
        scores = cross_val_score(ridge, X, y, cv=cv, scoring="r2")
        print(np.mean(scores), np.std(scores))
        r2_df = pd.concat([r2_df, pd.DataFrame({"type": [factor] * len(scores), "r2": scores, "target": [target] * len(scores)})]
    
r2_df["r2"] = np.clip(r2_df["r2"], 0, 1)
r2_df.to_csv("diffusion_neuromaps/data/ridge/r2_factor.csv", index=False)
