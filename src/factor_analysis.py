import numpy as np
import pandas as pd
import os
from factor_analyzer import FactorAnalyzer

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from tools import plot_glasser_map, plot_glasser_hemi


def compute_li(df: pd.DataFrame) -> pd.DataFrame:
    left_idx = [idx for idx in df.index if idx.endswith("_L")]
    right_idx = [idx for idx in df.index if idx.endswith("_R")]
    li_idx = [idx[:-2] for idx in left_idx]
    df_lh = df.loc[left_idx].values.ravel().astype(float)
    df_rh = df.loc[right_idx].values.ravel().astype(float)
    li_data = df_lh - df_rh
    li_df = pd.DataFrame(li_data, index=li_idx, columns=[df.columns[0] + "_li"])
    li_df.index.name = "region"
    return li_df


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

    df_li = compute_li(df)
    df_li.to_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "ave", f"F{i+1}_glasser_li.csv"))
    plot_glasser_hemi(df_li.values.ravel(), layout = ["lh", "lh"], view=["lateral", "medial"], size=(800, 400), 
    cmap="RdBu_r", label_text=None, color_bar=None, nan_color=(255, 255, 255, 1), share="row",
    color_range="sym", zoom=1.25, transparent_bg=False, interactive=False, 
    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", f"F{i+1}_glasser_li.png"))


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
fig.savefig("diffusion_neuromaps/plots/figs/glasser_factor_loadings.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)


subjects = np.loadtxt("diffusion_neuromaps/data/hcp_qc_subjects.txt", dtype=str)
metrics = ["fa", "md", "ad", "rd", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
              "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp"]
hcp_struct = np.zeros((len(metrics), len(subjects), 360))
for idx, metric in enumerate(metrics):
    hcp_struct[idx, :, :] = pd.read_csv(os.path.join(f"diffusion_neuromaps/data/glasser/subject/{metric}_glasser.csv"), index_col=0).values.astype(float)

f1_df = pd.DataFrame(index=subjects, columns=regions, dtype=float)
f2_df = pd.DataFrame(index=subjects, columns=regions, dtype=float)
f3_df = pd.DataFrame(index=subjects, columns=regions, dtype=float)
f4_df = pd.DataFrame(index=subjects, columns=regions, dtype=float)

for idx in range(len(subjects)):
    X = hcp_struct[:, idx, :].T
    X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
    scores = factor_analyzer.transform(X)
    f1_df.iloc[idx, :] = scores[:, 0]
    f2_df.iloc[idx, :] = scores[:, 1]
    f3_df.iloc[idx, :] = scores[:, 2]
    f4_df.iloc[idx, :] = scores[:, 3]

f1_df.index.name = "subject"
f2_df.index.name = "subject"
f3_df.index.name = "subject"
f4_df.index.name = "subject"

f1_df.to_csv("diffusion_neuromaps/data/glasser/subject/F1_glasser.csv")
f2_df.to_csv("diffusion_neuromaps/data/glasser/subject/F2_glasser.csv")
f3_df.to_csv("diffusion_neuromaps/data/glasser/subject/F3_glasser.csv")
f4_df.to_csv("diffusion_neuromaps/data/glasser/subject/F4_glasser.csv")
