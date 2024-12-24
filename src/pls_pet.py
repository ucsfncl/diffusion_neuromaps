### This script runs a PLS analysis between the structural metrics and PET maps
### Borrows heavily from https://github.com/netneurolab/hansen_receptors/blob/main/code/cognition.py
import os
import numpy as np
import pandas as pd
import pyls
from tools import make_glasser_nulls, plot_glasser_map
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


###  1. Load the structural metrics and PET maps

metric_path = "diffusion_neuromaps/data"
glasser_df = pd.read_csv("diffusion_neuromaps/atlases/glasser_parc.csv", index_col=0)
metrics = [
    "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
    "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"
    ]

dki_dti_metrics = ["dki-fa", "dki-ad", "dki-rd", "dki-md"]
dki_metrics = ["mk", "ak", "rk", "kfa", "mkt"]
noddi_metrics = ["icvf", "odi", "isovf"]
mapmri_metrics = ["msd", "qiv", "rtop", "rtap", "rtpp"]
struct_metrics = ["thick", "myl"]

struct_df = pd.DataFrame(index=glasser_df.index, columns=metrics, dtype=float)
for metric in metrics:
    hcp_df = pd.read_csv(os.path.join(metric_path, "glasser", "ave", f"{metric}_glasser_ave.csv"), index_col=0)
    struct_df[metric] = hcp_df

pet_names =["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
            "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
            "MOR", "KOR", "NET", "NMDA", "VAChT"]

pet_df = pd.DataFrame(index=glasser_df.index, columns=pet_names, dtype=float)
for name in pet_names:
    pet_df[name] = pd.read_csv(os.path.join(metric_path, "pet", f"{name}_glasser_ave.csv"), index_col=0)

###  2. Run the PLS analysis
nspins = 9999 
gen = make_glasser_nulls(nspins)
spins = np.hstack(gen.randomize(np.arange(180), np.arange(180, 360))).T

X = scipy.stats.zscore(struct_df.values, axis=0)
Y = scipy.stats.zscore(pet_df.values, axis=0)

pls_result = pyls.behavioral_pls(X, Y, n_boot=nspins, n_perm=nspins, permsamples=spins, test_split=0, seed=0)

###  3. Save and plot the scores
df = pd.DataFrame(pls_result["x_scores"][:, 0], index=glasser_df.index, columns=["struct_score"])
df.index.name = "region"
df.to_csv("diffusion_neuromaps/data/pls/pet_pls_struct_score_glasser_ave.csv")

df = pd.DataFrame(pls_result["y_scores"][:, 0], index=glasser_df.index, columns=["pet_score"])
df.index.name = "region"
df.to_csv("diffusion_neuromaps/data/pls/pet_pls_receptor_score_glasser_ave.csv")

plot_glasser_map(pls_result["x_scores"][:, 0], size=(1600, 400), cmap="RdBu_r", label_text=None, color_bar=False, color_range="sym",
nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False, 
screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", "pet_pls_struct_score.png"))

plot_glasser_map(pls_result["y_scores"][:, 0], size=(1600, 400), cmap="RdBu_r", label_text=None, color_bar=False, color_range="sym",
nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", "pet_pls_receptor_score.png"))

###  4. Plot and save the structural loadings with confidence intervals
xload = pyls.behavioral_pls(Y, X, n_boot=10000, n_perm=0, test_split=0)
df = pd.DataFrame(xload["y_loadings"][:, 0], index=metrics, columns=["struct_loading"])
df.index.name = "metric"
df.to_csv("diffusion_neuromaps/data/pls/pet_pls_struct_loadings.csv")

err = (xload["bootres"]["y_loadings_ci"][:, 0, 1]
      - xload["bootres"]["y_loadings_ci"][:, 0, 0]) / 2
sorted_idx = np.argsort(xload["y_loadings"][:, 0])

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 1, figsize=(3, 6), gridspec_kw={"wspace": 0.0, "hspace": 0.0})
ax.barh(range(len(metrics)), xload["y_loadings"][sorted_idx, 0], xerr=err[sorted_idx], color=["red" if xload["y_loadings"][sorted_idx[i], 0] > 0 else "blue" for i in range(len(sorted_idx))])
ax.set_yticks(range(len(metrics)), labels=[metrics[i].upper() for i in sorted_idx], fontsize=14)
ax.set_xlabel("Structural Loadings", fontsize=14)
fig.tight_layout()
plt.savefig("diffusion_neuromaps/plots/figs/pet_pls_struct_load.png", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

###  5. Plot the receptor loadings with confidence intervals
df = pd.DataFrame(pls_result["y_loadings"][:, 0], index=pet_names, columns=["receptor_loading"])
df.index.name = "receptor"
df.to_csv("diffusion_neuromaps/data/pls/pet_pls_receptor_loadings.csv")

err = (pls_result["bootres"]["y_loadings_ci"][:, 0, 1]
      - pls_result["bootres"]["y_loadings_ci"][:, 0, 0]) / 2
sorted_idx = np.argsort(pls_result["y_loadings"][:, 0])

fig, ax = plt.subplots(1, 1, figsize=(3, 6), gridspec_kw={"wspace": 0.0, "hspace": 0.0})
ax.barh(range(len(pet_names)), pls_result["y_loadings"][sorted_idx, 0], xerr=err[sorted_idx], color=["red" if pls_result["y_loadings"][sorted_idx[i], 0] > 0 else "blue" for i in range(len(sorted_idx))])
ax.set_yticks(range(len(pet_names)), labels=[pet_names[i] for i in sorted_idx], fontsize=14)
ax.set_xlabel("Receptor Loadings", fontsize=14)
fig.tight_layout()
plt.savefig("diffusion_neuromaps/plots/figs/pet_pls_receptor_load.png", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

###  6. Find the explained variance of the first latent variable and plot the explained variance of each latent variable
cv = pls_result["singvals"]**2 / np.sum(pls_result["singvals"]**2)
p = pls_result["permres"]["pvals"][0]

print(f"The first latent variable explains {cv[0]*100:.2f}% of the covariance, with a p-value of {p:.5f}")

plt.figure(figsize=(4, 4))
plt.scatter(range(len(metrics)), cv*100, s=80)
plt.ylabel("percent covariance accounted for")
plt.xlabel("latent variable")
plt.title('PLS' + str(0) + ' cov exp = ' + str(cv[0])[:5]
          + ', pspin = ' + str(p)[:5])
plt.tight_layout()
plt.savefig("diffusion_neuromaps/plots/figs/pet_pls_explained_var.png", dpi=400, bbox_inches="tight", pad_inches=0.05)