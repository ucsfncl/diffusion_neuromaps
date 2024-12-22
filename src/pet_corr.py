import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from tools import df_pval, plot_glasser_map, compute_multilinear, compute_cv


###  1. Load the microstructure metrics
metric_path = os.path.join("diffusion_neuromaps", "data")
glasser_df = pd.read_csv(os.path.join("diffusion_neuromaps", "atlases", "glasser_parc.csv"), index_col=0)
metrics = [
    "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
    "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]
struct_df = pd.DataFrame(index=glasser_df.index, columns=metrics, dtype=float)
for metric in metrics:
    hcp_val = pd.read_csv(os.path.join(metric_path, "glasser", "ave", f"{metric}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
    hcp_val = (hcp_val - hcp_val.mean()) / hcp_val.std()
    struct_df[metric] = hcp_val


###  2. Load the PET receptor/transporter density maps
pet_names =["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
            "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
            "MOR", "KOR", "NET", "NMDA", "VAChT"]
pet_df = pd.DataFrame(index=glasser_df.index, columns=pet_names, dtype=float)
for name in pet_names:
    pet_val = pd.read_csv(os.path.join(metric_path, "pet", f"{name}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
    pet_val = (pet_val - pet_val.mean()) / pet_val.std()
    pet_df[name] = pet_val
    plot_glasser_map(pet_val, size=(1600, 400), cmap="Reds", label_text=None, color_bar=False, 
    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False, 
    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", f"{name}_glasser_ave.png"))


###  3. Perform univariate analysis and plot the correlation heatmap
p_pet, r_pet = df_pval(pet_df, struct_df)
r_pet[p_pet > 0.05] = np.nan
xlabels = [metric.upper() for metric in metrics]
ylabels = pet_names

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 1, figsize=(12, 14), gridspec_kw={"wspace": 0.0, "hspace": 0.0})
ax = sns.heatmap(data=r_pet.values, vmin=-1, vmax=1, ax=ax, cbar=True, square=True, cmap="RdBu_r", linewidths=.5,
xticklabels=xlabels, yticklabels=ylabels, cbar_kws={"format": "%.1f", "ticks": [-1, -0.5, 0, 0.5, 1], "shrink": 0.5, "pad": 0.01})
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.add_patch(patches.Rectangle((0.0, 0.0), 4, r_pet.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((4.0, 0.0), 5, r_pet.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((9.0, 0.0), 3, r_pet.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((12.0, 0.0), 5, r_pet.shape[0], fill=False, edgecolor='black', lw=1))
ax.add_patch(patches.Rectangle((17.0, 0.0), 2, r_pet.shape[0], fill=False, edgecolor='black', lw=1))
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
fig.savefig(os.path.join("diffusion_neuromaps", "plots", "figs", "pet_corr.png"), dpi=400, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)


### 4. Perform multiple regression analysis
r_df, p_df, pet_pred = compute_multilinear(struct_df, pet_df)
train_r, test_r = compute_cv(struct_df, pet_df)
for name in pet_names:
    print(f"Train {name}: {np.mean(train_r[name])}, {np.percentile(train_r[name], 5)}, {np.percentile(train_r[name], 95)}")
    print(f"Test {name}: {np.mean(test_r[name])}, {np.percentile(test_r[name], 5)}, {np.percentile(test_r[name], 95)}")

r_pred = {name: np.corrcoef(pet_pred[:, idx], pet_df[name].values.ravel().astype(float))[0, 1] for idx, name in enumerate(pet_names)}
### 5. What percentage of multivariate prediction is explained by the Mesulam parcels?
mesulam_val = glasser_df["mesulam"].values.ravel()
mesulam_regions = np.unique(mesulam_val)
mesulam = np.zeros((360, 4))
for i, region in enumerate(mesulam_regions):
    mesulam[mesulam_val == region, i] = 1
w_mesulam = np.linalg.lstsq(mesulam, pet_pred, rcond=None)[0]
mesulam_pred = mesulam.dot(w_mesulam)

r_mesulam = {name: np.corrcoef(mesulam_pred[:, idx], pet_pred[:, idx])[0, 1] for idx, name in enumerate(pet_names)}

### 6. Plot the results
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex="all")
ax[0].bar(r_pred.keys(), r_pred.values(), color="blue")
ax[0].set_ylabel("Multivariate Correlation (R)")
ax[0].set_title("A", loc="left", fontsize=14, fontweight="bold")
ax[1].bar(r_mesulam.keys(), r_mesulam.values(), color="blue")
ax[1].set_ylabel("% Variance Explained by Mesulam")
ax[1].set_title("B", loc="left", fontsize=14, fontweight="bold")
ax[2] = sns.boxplot(data=test_r,showfliers=False, ax=ax[2], color="blue")
ax[2].set_ylabel("Test CV Correlation (R)")
ax[2].set_xticks(np.arange(len(pet_names)), labels=pet_names, rotation=270)
ax[2].set_xlabel("")
ax[2].set_title("C", loc="left", fontsize=14, fontweight="bold")
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[2].spines["top"].set_visible(False)
ax[2].spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("diffusion_neuromaps/plots/figs/pet_cv.png", dpi=400, pad_inches=0.05)
plt.close(fig)