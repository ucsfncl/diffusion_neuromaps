# Plots the dominancel analysis results
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from tools import compute_dominance
import seaborn as sns
import matplotlib.patches as patches
top_k = 20


### 1. Load and plot the MEG dominance data 
metrics = [
    "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
    "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"
    ]
meg_names = ["megdelta", "megtheta", "megalpha", "megbeta", "meggamma1", "meggamma2", "megtimescale"]
meg_res = pd.DataFrame(index=meg_names, columns=metrics, dtype=float)
for name in meg_names:
    meg_df = pd.read_csv(f"diffusion_neuromaps/data/dominance/meg_{name}_dominance.csv", index_col=0)
    meg_res.loc[name, :] = meg_df.values


xlabels = [metric.upper() for metric in metrics]
ylabels = ["Delta", "Theta", "Alpha", "Beta", "Low Gamma", "High Gamma", "Timescale"]

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 1, figsize=(12, 6), gridspec_kw={"wspace": 0.0, "hspace": 0.0})
ax = sns.heatmap(data=meg_res.values, ax=ax, cbar=True, square=True, cmap="Reds", linewidths=.5, vmin=0.0, vmax=np.nanmax(meg_res.values),
xticklabels=xlabels, yticklabels=ylabels, cbar_kws={"shrink": 0.5, "pad": 0.01, "ticks": [0.0, 5.0, 10.0, 15.0, 20.0]})
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.add_patch(patches.Rectangle((0.0, 0.0), 4, meg_res.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((4.0, 0.0), 5, meg_res.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((9.0, 0.0), 3, meg_res.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((12.0, 0.0), 5, meg_res.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((17.0, 0.0), 2, meg_res.shape[0], fill=False, edgecolor='black', lw=1))
for i in range(4):
    ax.get_xticklabels()[i].set_color("purple")
for i in range(4, 9):
    ax.get_xticklabels()[i].set_color("orange")
for i in range(9, 12):
    ax.get_xticklabels()[i].set_color("blue")
for i in range(12, 17):
    ax.get_xticklabels()[i].set_color("green")
for i in range(17, 19):
    ax.get_xticklabels()[i].set_color("black")
fig.tight_layout()
fig.savefig(os.path.join("diffusion_neuromaps", "plots", "figs", "meg_dominance.png"), dpi=400, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)


###  2. Load and plot the PET dominance data
pet_names =["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
            "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
            "MOR", "KOR", "NET", "NMDA", "VAChT"]

pet_res = pd.DataFrame(index=pet_names, columns=metrics, dtype=float)
for name in pet_names:
    pet_df = pd.read_csv(f"diffusion_neuromaps/data/dominance/pet_{name}_dominance.csv", index_col=0)
    pet_res.loc[name, :] = pet_df.values

xlabels = [metric.upper() for metric in metrics]
ylabels = pet_names
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 1, figsize=(8, 8), gridspec_kw={"wspace": 0.0, "hspace": 0.0})
ax = sns.heatmap(data=pet_res.values, ax=ax, cbar=True, square=True, cmap="Reds", linewidths=.5, vmin=0.0, vmax=np.nanmax(pet_res.values),
xticklabels=xlabels, yticklabels=ylabels, cbar_kws={"shrink": 0.5, "pad": 0.01, "ticks": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]})
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.add_patch(patches.Rectangle((0.0, 0.0), 4, pet_res.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((4.0, 0.0), 5, pet_res.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((9.0, 0.0), 3, pet_res.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((12.0, 0.0), 5, pet_res.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((17.0, 0.0), 2, pet_res.shape[0], fill=False, edgecolor='black', lw=1))
for i in range(4):
    ax.get_xticklabels()[i].set_color("purple")
for i in range(4, 9):
    ax.get_xticklabels()[i].set_color("orange")
for i in range(9, 12):
    ax.get_xticklabels()[i].set_color("blue")
for i in range(12, 17):
    ax.get_xticklabels()[i].set_color("green")
for i in range(17, 19):
    ax.get_xticklabels()[i].set_color("black")
fig.tight_layout()
fig.savefig(os.path.join("diffusion_neuromaps", "plots", "figs", "pet_dominance.png"), dpi=400, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)


###  3. Load and plot the disorder dominance data
metrics = ["dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt", "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "myl"]
disorder_names = ["adhd_ct", "adhd_sa", "bd_ct", "bd_sa", "mdd_ct", "mdd_sa", "ocd_ct", "ocd_sa", "scz_ct", "scz_sa", "asd_ct"]
disorder_res = pd.DataFrame(index=disorder_names, columns=metrics, dtype=float)
for name in disorder_names:
    disorder_df = pd.read_csv(f"diffusion_neuromaps/data/dominance/disorder_{name}_dominance.csv", index_col=0)
    disorder_res.loc[name, :] = disorder_df.values

max_val = np.nanmax(disorder_res.values)
val_ticks = np.arange(0.0, max_val, 5.0)
xlabels = [metric.upper() for metric in metrics]
ylabels = ["ADHD CT", "ADHD SA", "ASD CT", "BD CT", "BD SA", "MDD CT", "MDD SA", "OCD CT", "OCD SA", "SCZ CT", "SCZ SA"]
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 1, figsize=(12, 12), gridspec_kw={"wspace": 0.0, "hspace": 0.0})
ax = sns.heatmap(data=disorder_res.values, ax=ax, cbar=True, square=True, cmap="Reds", linewidths=.5, vmin=0.0, vmax=np.nanmax(disorder_res.values),
xticklabels=xlabels, yticklabels=ylabels, cbar_kws={"shrink": 0.5, "pad": 0.01, "ticks": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]})
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.add_patch(patches.Rectangle((0.0, 0.0), 4, disorder_res.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((4.0, 0.0), 5, disorder_res.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((9.0, 0.0), 3, disorder_res.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((12.0, 0.0), 5, disorder_res.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((17.0, 0.0), 1, disorder_res.shape[0], fill=False, edgecolor='black', lw=1))
for i in range(4):
    ax.get_xticklabels()[i].set_color("purple")
for i in range(4, 9):
    ax.get_xticklabels()[i].set_color("orange")
for i in range(9, 12):
    ax.get_xticklabels()[i].set_color("blue")
for i in range(12, 17):
    ax.get_xticklabels()[i].set_color("green")
fig.tight_layout()
fig.savefig(os.path.join("diffusion_neuromaps", "plots", "figs", "disorder_dominance.png"), dpi=400, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)