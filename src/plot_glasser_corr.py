from tools import plot_glasser_map, plot_glasser_hemi
import os
import numpy as np
import pandas as pd

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from tools import spin_test_glasser, make_glasser_nulls
import scipy
import scipy.stats


### 1. Load the structural metrics
metrics = [
    "fa", "ad", "rd", "md", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
    "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"
    ]
dti_metrics = ["fa", "ad", "rd", "md"]
dki_dti_metrics = ["dki-fa", "dki-ad", "dki-rd", "dki-md"]
dki_metrics = ["mk", "ak", "rk", "kfa", "mkt"]
noddi_metrics = ["icvf", "odi", "isovf"]
mapmri_metrics = ["msd", "qiv", "rtop", "rtap", "rtpp"]
struct_metrics = ["thick", "myl"]

hcp_struct = np.zeros((len(metrics), 360))
for idx, metric in enumerate(metrics):
    hcp_struct[idx, :] = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "ave", f"{metric}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)

r_mat = np.corrcoef(hcp_struct)
np.fill_diagonal(r_mat, 0.0)
triu_indices = np.triu_indices(len(metrics), k=1)
tril_indices = np.tril_indices(len(metrics), k=-1)

### 2. Spin test
n_rep = 9999
gen = make_glasser_nulls(n_rep)
pvals = np.zeros(r_mat.shape)
for idx, metric in enumerate(metrics):
    for j in range(idx):
        r, null_r, p = spin_test_glasser(gen, hcp_struct[idx, :], hcp_struct[j, :])
        print(f"{metric} vs {metrics[j]}: {r_mat[idx, j]:.3f} p: {p:.4f}")
        pvals[idx, j] = p
        pvals[j, idx] = p

pvals = pvals[tril_indices]
pvals = scipy.stats.false_discovery_control(pvals, method="bh")
unsig_mask = np.zeros(r_mat.shape, dtype=bool)
unsig_mask[tril_indices] = pvals > 0.05
r_mat[unsig_mask] = np.NaN
np.fill_diagonal(r_mat, np.NaN)

# ### 3. Plot the correlation matrix
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 1, figsize=(8, 8), gridspec_kw={"hspace": 0.0, "wspace": 0.0})
ax = sns.heatmap(data=r_mat, vmin=-1, vmax=1, ax=ax, cbar=True, square=True, cmap="RdBu_r",
            linewidths=.5, xticklabels=[metric.upper() for metric in metrics], yticklabels=[metric.upper() for metric in metrics],
            cbar_kws={"format": "%.1f", "ticks": [-1, -0.5, 0, 0.5, 1], "shrink": 0.5, "pad": 0.01})
# set all unsig to white
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.add_patch(patches.Rectangle((0.0, 0.0), 4, 4, fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((4.0, 4.0), 4, 4, fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((8.0, 8.0), 5, 5, fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((13.0, 13.0), 3, 3, fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((16.0, 16.0), 5, 5, fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((21.0, 21.0), 2, 2, fill=False, edgecolor='black', lw=1))
for i in range(4):
    ax.get_xticklabels()[i].set_color("red")
    ax.get_yticklabels()[i].set_color("red")
for i in range(4, 8):
    ax.get_xticklabels()[i].set_color("purple")
    ax.get_yticklabels()[i].set_color("purple")
for i in range(8, 13):
    ax.get_xticklabels()[i].set_color("orange")
    ax.get_yticklabels()[i].set_color("orange")
for i in range(13, 16):
    ax.get_xticklabels()[i].set_color("blue")
    ax.get_yticklabels()[i].set_color("blue")
for i in range(16, 21):
    ax.get_xticklabels()[i].set_color("green")
    ax.get_yticklabels()[i].set_color("green")
for i in range(21, 23):
    ax.get_xticklabels()[i].set_color("black")
    ax.get_yticklabels()[i].set_color("black")
ax.set_title("Spatial Correlation")
fig.tight_layout()
plt.savefig("diffusion_neuromaps/plots/figs/glasser_ave_corr.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
plt.close()
