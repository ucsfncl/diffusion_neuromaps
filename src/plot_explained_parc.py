# Plot explained variation of structural metrics by different parcellations
import numpy as np
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

def r2(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def explained_parc(y, parc):
    y = (y - np.mean(y)) / np.std(y)
    X = np.zeros((y.shape[0], np.max(parc)))
    for i in range(np.max(parc)):
        X[parc == i + 1, i] = 1
    
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ w
    return adjusted_r2(r2(y, y_pred), y.shape[0], X.shape[1])


### 1. Load the structural metrics and parcellations
metric_path = "diffusion_neuromaps/data"
glasser_df = pd.read_csv("diffusion_neuromaps/atlases/glasser_parc.csv", index_col=0)
metrics = ["fa", "ad", "rd", "md", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
            "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]

struct_df = pd.DataFrame(index=glasser_df.index, columns=metrics, dtype=float)
for metric in metrics:
    hcp_df = pd.read_csv(os.path.join(metric_path, "glasser", "ave", f"{metric}_glasser_ave.csv"), index_col=0)
    struct_df[metric] = (hcp_df - hcp_df.mean()) / hcp_df.std()

sa_axis = pd.read_csv(os.path.join(metric_path, "neuromaps", "SAaxis_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
mesulam = glasser_df["mesulam"].map({"paralimbic": 1, "heteromodal": 2, "unimodal": 3, "idiotypic": 4}).values.ravel().astype(int)
economo = glasser_df["economo"].map({"agranular": 1, "frontal": 2, "parietal": 3, "polar": 4, "granular": 5}).values.ravel().astype(int)
yeo = glasser_df["yeo"].map({"visual": 1, "somatosensory": 2, "dorsal attention": 3, "ventral attention": 4, "limbic": 5, "frontoparietal": 6, "default mode": 7}).values.ravel().astype(int)


### 2. Calculate and plot the explained variation of the structural metrics by different parcellations
sa_exp = []
mes_exp = []
yeo_exp = []
eco_exp = []

for metric in metrics:
    sa_exp.append(np.corrcoef(struct_df[metric].values, sa_axis)[0, 1] ** 2 * 100)
    mes_exp.append(explained_parc(struct_df[metric].values, mesulam) * 100)
    yeo_exp.append(explained_parc(struct_df[metric].values, yeo) * 100)
    eco_exp.append(explained_parc(struct_df[metric].values, economo) * 100)

palette = {"fa": "red", "ad": "red", "rd": "red", "md": "red",
    "dki-fa": "purple", "dki-ad": "purple", "dki-rd": "purple", "dki-md": "purple", 
    "mk": "orange", "ak": "orange", "rk": "orange", "kfa": "orange", "mkt": "orange",
    "icvf": "blue", "odi": "blue", "isovf": "blue", "msd": "green", "qiv": "green", "rtop": "green", "rtap": "green", "rtpp": "green", 
    "thick": "black", "myl": "black"}

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(4, 1, figsize=(16, 20), sharex="all", sharey="all", gridspec_kw={"wspace": 0.0, "hspace": 0.1})
ax[0] = sns.barplot(x=metrics, y=sa_exp, ax=ax[0], palette=palette)
ax[0].set_ylabel("SA Axis", fontsize=14)
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[1] = sns.barplot(x=metrics, y=mes_exp, ax=ax[1], palette=palette)
ax[1].set_ylabel("Mesulam", fontsize=14)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[2] = sns.barplot(x=metrics, y=yeo_exp, ax=ax[2], palette=palette)
ax[2].set_ylabel("Yeo Networks", fontsize=14)
ax[2].spines["top"].set_visible(False)
ax[2].spines["right"].set_visible(False)
ax[3] = sns.barplot(x=metrics, y=eco_exp, ax=ax[3], palette=palette)
ax[3].set_ylabel("Economo-Koskinas", fontsize=14)
ax[3].spines["top"].set_visible(False)
ax[3].spines["right"].set_visible(False)
ax[3].set_xticks(range(len(metrics)), [metric.upper() for metric in metrics], rotation=270, fontsize=14)
for i in range(4):
    ax[3].get_xticklabels()[i].set_color("red")
for i in range(4, 8):
    ax[3].get_xticklabels()[i].set_color("purple")
for i in range(8, 13):
    ax[3].get_xticklabels()[i].set_color("orange")
for i in range(13, 16):
    ax[3].get_xticklabels()[i].set_color("blue")
for i in range(16, 21):
    ax[3].get_xticklabels()[i].set_color("green")
fig.tight_layout()
fig.savefig("diffusion_neuromaps/plots/figs/explained_parc_variation.png", bbox_inches="tight", dpi=400, pad_inches=0.05)
plt.close(fig)

