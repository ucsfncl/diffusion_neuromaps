# Loads the HCP surface data and calculate the average value of each Glasser region
import numpy as np
import pandas as pd
import os
import shutil
from factor_analyzer import FactorAnalyzer
from tools import plot_map


from nilearn.surface import load_surf_data
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

# data_path = "/protected/data/ncl-mb12/HCP_SURFACE_RETEST"
# output_path = "/protected/data/ncl-mb12/brain_maps/glasser_retest"
# subjects = np.loadtxt("microstructure/hcp_retest_qc_subjects.txt", dtype=str).tolist()


metrics = ["fa", "md", "ad", "rd", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
           "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp"]

hcp_df = pd.DataFrame(index=metrics, columns=glasser_df.index.to_list())
for metric in metrics:
    df = pd.read_csv(os.path.join(f"diffusion_neuromaps/{metric}_glasser_ave.csv"), index_col=0)
    hcp_df.loc[metric] = df["0"]

print(hcp_df.head())

X = hcp_df.values.T
X = np.array(X, dtype=float)
print(X.shape)
X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)

fa_method = "minres" # minimum residuals FA
rotation = "promax" # oblique rotation, factors should be allowed to correlate with each other
n_factor_select = "factor_ev" # select factor number based on total or factor-wise EV
n_factor_thresh = 0.01 # retain factors that explain > 1% of variance

fa = FactorAnalyzer(n_factors=X.shape[1], method=fa_method, rotation=None, svd_method="lapack")
fa.fit(X)
# get eigenvalues & explained variance
eig,_ = fa.get_eigenvalues()
_, ev_frac, ev = fa.get_factor_variance()

# select n_factors based on overall explained variance
if n_factor_select=="total_ev":
    n_factors = [i for i in range(len(ev)) if (ev[i] >= n_factor_thresh)][1]
    print(f"n = {n_factors} factors explain >= {n_factor_thresh*100}% of variance")
# select n_factors based on factor-level explained variance
elif n_factor_select=="factor_ev":
    n_factors = [i+1 for i in range(len(ev_frac)) if (ev_frac[i] >= n_factor_thresh)][-1]
    print(f"n = {n_factors} factors with ev >= {n_factor_thresh*100}%")

# n_factors = 5
fa = FactorAnalyzer(n_factors=n_factors, method=fa_method, rotation=rotation, svd_method="lapack")

n_factors = 4
fa.fit(X)
# get loadings and derive factor names
loadings = pd.DataFrame(fa.loadings_[:,:n_factors], index=hcp_df.index, columns=[f"ni{i+1}" for i in range(n_factors)])
print(loadings)
labels = dict()
pred_factor_idx = np.array([loadings.abs().loc[p,:].idxmax() for p in hcp_df.index])
pred_factor_val = np.array([loadings.abs().loc[p,:].max() for p in hcp_df.index])

r_loadings = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        r_loadings[i, j] = np.corrcoef(loadings.iloc[:, i], loadings.iloc[:, j])[0, 1]

print(r_loadings)

print(pred_factor_idx)
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni1")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni2")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni3")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni4")[0]])

scores = fa.transform(X)
print(np.mean(scores, axis=0))
print(np.std(scores, axis=0))

for i in range(4):
    df = pd.DataFrame(scores[:, i], index=glasser_df.index, columns=[f"F{i+1}"])
    df.to_csv(os.path.join(output_path, f"F{i+1}_glasser_ave.csv"))

print(scores.shape)
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
# ax = sns.heatmap(data=meg_res.values, ax=ax, cbar=True, square=True, cmap="Reds", linewidths=.5, vmin=0.0, vmax=np.nanmax(meg_res.values),
# xticklabels=xlabels, yticklabels=ylabels, cbar_kws={"shrink": 0.5, "pad": 0.01, "ticks": [0.0, 5.0, 10.0, 15.0, 20.0]})
ax = sns.heatmap(loadings.values.T, ax=ax, cmap="RdBu_r", square=True, 
yticklabels=[f"F{i+1}" for i in range(n_factors)], xticklabels=[metric.upper() for metric in metrics], cbar=False)
                    # cbar_kws={"shrink": 0.4, "pad": 0.01, "ticks": [-1.0, 0.0, 1.0]}, vmin=-1.0, vmax=1.0, linewidths=.5)
for i in range(loadings.shape[0]):
    j_idx = np.argmax(np.abs(loadings.iloc[i, :]))
    ax.add_patch(patches.Rectangle((i, j_idx), 1, 1, fill=False, edgecolor="black", lw=1))
ax.set_xticklabels([metric.upper() for metric in metrics], fontsize=16)
ax.set_yticklabels([f"F{i+1}" for i in range(n_factors)], fontsize=16, rotation=0)

for i in range(4):
    ax.get_xticklabels()[i].set_color("red")
for i in range(4, 8):
    ax.get_xticklabels()[i].set_color("purple")
for i in range(8, 13):
    ax.get_xticklabels()[i].set_color("orange")
for i in range(13, 16):
    ax.get_xticklabels()[i].set_color("blue")
for i in range(16, 21):
    ax.get_xticklabels()[i].set_color("green")

fig.tight_layout()
fig.savefig("microstructure/paper_figs/glasser_loadings.png", dpi=400, bbox_inches="tight", pad_inches=0.05)

for i in range(4):
    plot_map(scores[:, i], None, f"microstructure/paper_figs/F{i+1}_scores.png")
