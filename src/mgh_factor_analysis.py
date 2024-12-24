import numpy as np
import pandas as pd
import os
import shutil
from factor_analyzer import FactorAnalyzer

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

from tools import plot_glasser_map


###  1. Perform factor analysis on the MGH data (full)
metrics = ["fa", "md", "ad", "rd", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
           "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp"]
glasser_df = pd.read_csv("diffusion_neuromaps/atlases/glasser_parc.csv", index_col=0)
mgh_df = pd.DataFrame(index=glasser_df.index, columns=metrics, dtype=float)
for metric in metrics:
    df = pd.read_csv(f"diffusion_neuromaps/data/mgh/{metric}_glasser_ave.csv", index_col=0).values.ravel().astype(float)
    mgh_df[metric] = df

X = mgh_df.values
X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)

factor_analyzer = FactorAnalyzer(n_factors=6, method="minres", rotation="promax", svd_method="lapack")
factor_analyzer.fit(X)
n_factors = 4
loadings = pd.DataFrame(factor_analyzer.loadings_[:,:n_factors], index=metrics, columns=[f"ni{i+1}" for i in range(n_factors)])
pred_factor_idx = np.array([loadings.abs().loc[p, :].idxmax() for p in metrics])
pred_factor_val = np.array([loadings.abs().loc[p, :].max() for p in metrics])

print([metrics[idx] for idx in np.where(pred_factor_idx=="ni1")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni2")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni3")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni4")[0]])

scores = factor_analyzer.transform(X)
for i in range(n_factors):
    df = pd.DataFrame(scores[:, i], index=glasser_df.index, columns=[f"F{i+1}"])
    df.index.name = "region"
    df.to_csv(os.path.join("diffusion_neuromaps", "data", "mgh", f"F{i+1}_glasser_ave.csv"))
    plot_glasser_map(scores[:, i], size=(1600, 400), cmap="Reds", label_text=None, color_bar=False,
    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", f"F{i+1}_glasser_ave.png"))


fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax = sns.heatmap(loadings.values.T[:n_factors, :], ax=ax, cmap="RdBu_r", square=True, yticklabels=[f"F{i+1}" for i in range(n_factors)], xticklabels=[metric.upper() for metric in metrics], cbar=False)
for i in range(len(metrics)):
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
fig.savefig("diffusion_neuromaps/plots/figs/mgh_glasser_factor_loadings.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

###  2. Perform factor analysis on the MGH data (limited)
metrics = ["fa", "md", "ad", "rd", "dki-fa-limited", "dki-ad-limited", "dki-rd-limited", "dki-md-limited", "mk-limited", "ak-limited", "rk-limited", "kfa-limited", "mkt-limited",
           "icvf-limited", "odi-limited", "isovf-limited", "msd-limited", "qiv-limited", "rtop-limited", "rtap-limited", "rtpp-limited"]
mgh_df = pd.DataFrame(index=glasser_df.index, columns=metrics, dtype=float)
for metric in metrics:
    df = pd.read_csv(f"diffusion_neuromaps/data/mgh/{metric}_glasser_ave.csv", index_col=0).values.ravel().astype(float)
    mgh_df[metric] = df

X = mgh_df.values
X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)

fa = FactorAnalyzer(n_factors=5, method="minres", rotation="promax", svd_method="lapack")
n_factors = 4
fa.fit(X)
loadings = pd.DataFrame(fa.loadings_[:,:n_factors], index=metrics, columns=[f"ni{i+1}" for i in range(n_factors)])
loadings = loadings[["ni2", "ni1", "ni4", "ni3"]]
pred_factor_idx = np.array([loadings.abs().loc[p, :].idxmax() for p in metrics])
pred_factor_val = np.array([loadings.abs().loc[p, :].max() for p in metrics])

print([metrics[idx] for idx in np.where(pred_factor_idx=="ni1")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni2")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni3")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni4")[0]])

scores = factor_analyzer.transform(X)
scores = scores[:, [1, 0, 3, 2]]
for i in range(n_factors):
    df = pd.DataFrame(scores[:, i], index=glasser_df.index, columns=[f"F{i+1}"])
    df.index.name = "region"
    df.to_csv(os.path.join("diffusion_neuromaps", "data", "mgh", f"F{i+1}-limited_glasser_ave.csv"))
    plot_glasser_map(scores[:, i], size=(1600, 400), cmap="Reds", label_text=None, color_bar=False,
    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", f"F{i+1}-limited_glasser_ave.png"))


fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax = sns.heatmap(loadings.values.T[:n_factors, :], ax=ax, cmap="RdBu_r", square=True, yticklabels=[f"F{i+1}" for i in range(n_factors)], xticklabels=[metric.upper() for metric in metrics], cbar=False)
for i in range(len(metrics)):
    j_idx = np.argmax(np.abs(loadings.iloc[i, :]))
    ax.add_patch(patches.Rectangle((i, j_idx), 1, 1, fill=False, edgecolor="black", lw=1))
ax.set_xticklabels(["-".join(metric.split("-")[:-1]).upper() for metric in metrics], fontsize=16)
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
fig.savefig("diffusion_neuromaps/plots/figs/mgh-limited_glasser_factor_loadings.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
