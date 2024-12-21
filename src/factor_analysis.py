import numpy as np
import pandas as pd
import os
from factor_analyzer import FactorAnalyzer

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from tools import plot_glasser_map


metrics = ["fa", "md", "ad", "rd", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
           "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp"]
glasser_df = pd.read_table("diffusion_neuromaps/atlases/Glasser360_Atlas.txt", sep=",", index_col=0, header=0, encoding="utf-8")

regions = glasser_df.index.tolist()
hcp_struct = np.zeros((len(metrics), 360))
for idx, metric in enumerate(metrics):
    hcp_struct[idx, :] = pd.read_csv(os.path.join(f"diffusion_neuromaps/data/glasser/ave/{metric}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)

X = hcp_struct.T
X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)

factor_analyzer = FactorAnalyzer(n_factors=X.shape[1], method="minres", rotation=None, svd_method="lapack")
factor_analyzer.fit(X)
_, ev_frac, ev = factor_analyzer.get_factor_variance()
n_factors = (ev_frac >= 0.01).sum()

print(f"n = {n_factors} factors with ev >= 1%")
factor_analyzer = FactorAnalyzer(n_factors=n_factors, method="minres", rotation="promax", svd_method="lapack")
factor_analyzer.fit(X)
loadings = pd.DataFrame(factor_analyzer.loadings_[:,:n_factors], index=metrics, columns=[f"ni{i+1}" for i in range(n_factors)])
pred_factor_idx = np.array([loadings.abs().loc[p, :].idxmax() for p in metrics])
pred_factor_val = np.array([loadings.abs().loc[p, :].max() for p in metrics])

print([metrics[idx] for idx in np.where(pred_factor_idx=="ni1")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni2")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni3")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni4")[0]])
print([metrics[idx] for idx in np.where(pred_factor_idx=="ni5")[0]])
### Factor 5 has no MAX loadings so it is not considered
n_factors = 4
scores = factor_analyzer.transform(X)
for i in range(n_factors):
    df = pd.DataFrame(scores[:, i], index=regions, columns=[f"F{i+1}"])
    df.index.name = "region"
    df.to_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "ave", f"F{i+1}_glasser_ave.csv"))
    plot_glasser_map(scores[:, i], size=(1600, 400), cmap="Reds", label_text=None, color_bar=False,
    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", f"F{i+1}_glasser_ave.png"))

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax = sns.heatmap(loadings.values.T[:n_factors, :], ax=ax, cmap="RdBu_r", square=True, yticklabels=[f"F{i+1}" for i in range(n_factors)], xticklabels=[metric.upper() for metric in metrics], cbar=False)
for i in range(n_factors):
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
fig.savefig("diffusion_neuromaps/plots/figs/glasser_factor_loadings.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)