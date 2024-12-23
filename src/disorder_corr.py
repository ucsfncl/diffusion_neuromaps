import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import json

from tools import compute_multilinear, compute_cv, plot_dk_map


###  1. Load the microstructure metrics and case-control maps
metric_path = os.path.join("diffusion_neuromaps", "data")
metrics = [
    "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
    "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "myl"]

with open("diffusion_neuromaps/atlases/fslr_dk.json", "r") as f:
    fslr_labels = json.load(f)
fslr_labels = {int(k): v for k, v in fslr_labels.items()}
dk_regions = [f"lh.{val}" for val in fslr_labels.values()] + [f"rh.{val}" for val in fslr_labels.values()]

struct_df = pd.DataFrame(index=dk_regions, columns=metrics, dtype=float)
for metric in metrics:
    hcp_df = pd.read_csv(os.path.join(metric_path, "dk", "ave", f"{metric}_dk_ave.csv"), index_col=0)
    plot_dk_map(hcp_df, size=(1600, 400), cmap="Reds", label_text=None, color_bar=False,
                    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
                    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "dk", f"{metric}_dk_ave.png"))
    struct_df[metric] = hcp_df

ct_disorders = ["adhd", "bd", "mdd", "ocd", "scz", "asd"]
sa_disorders = ["adhd", "bd", "mdd", "ocd", "scz"]
ct_df = pd.DataFrame(index=dk_regions, columns=ct_disorders, dtype=float)
sa_df = pd.DataFrame(index=dk_regions, columns=sa_disorders, dtype=float)
for disorder in ct_disorders:
    ct_df[disorder] = pd.read_csv(os.path.join(metric_path, "disorders", f"{disorder}_ct_dk_ave.csv"), index_col=0)
    plot_dk_map(ct_df[disorder], size=(1600, 400), cmap="RdBu_r", label_text=None, color_bar=False, color_range="sym",
                    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
                    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "dk", f"{disorder}_ct_dk_ave.png"))

for disorder in sa_disorders:
    sa_df[disorder] = pd.read_csv(os.path.join(metric_path, "disorders", f"{disorder}_sa_dk_ave.csv"), index_col=0)
    plot_dk_map(sa_df[disorder], size=(1600, 400), cmap="RdBu_r", label_text=None, color_bar=False, color_range="sym",
                    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
                    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "dk", f"{disorder}_sa_dk_ave.png"))



###  2. Perform multiple regression analysis
r_ct, p_ct, ct_pred = compute_multilinear(struct_df, ct_df)
r_sa, p_sa, sa_pred = compute_multilinear(struct_df, sa_df)
train_r_ct, test_r_ct = compute_cv(struct_df, ct_df)
train_r_sa, test_r_sa = compute_cv(struct_df, sa_df)
r_ct_pred = {name: np.corrcoef(ct_pred[:, idx], ct_df[name].values.ravel().astype(float))[0, 1] for idx, name in enumerate(ct_disorders)}
r_sa_pred = {name: np.corrcoef(sa_pred[:, idx], sa_df[name].values.ravel().astype(float))[0, 1] for idx, name in enumerate(sa_disorders)}
for name in ct_disorders:
    print(f"Train {name} CT: {np.mean(train_r_ct[name])}, {np.percentile(train_r_ct[name], 5)}, {np.percentile(train_r_ct[name], 95)}")
    print(f"Test {name} CT: {np.mean(test_r_ct[name])}, {np.percentile(test_r_ct[name], 5)}, {np.percentile(test_r_ct[name], 95)}")
for name in sa_disorders:
    print(f"Train {name} SA: {np.mean(train_r_sa[name])}, {np.percentile(train_r_sa[name], 5)}, {np.percentile(train_r_sa[name], 95)}")
    print(f"Test {name} SA: {np.mean(test_r_sa[name])}, {np.percentile(test_r_sa[name], 5)}, {np.percentile(test_r_sa[name], 95)}")

###  3. Plot the results
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(4, 1, figsize=(8, 12), sharex="all", gridspec_kw={"wspace": 0.0, "hspace": 0.3})
ax[0].bar(r_sa_pred.keys(), r_sa_pred.values(), color="blue")
ax[0].set_ylabel("Multivariate Surface Area (R)")
ax[0].set_title("A", loc="left", fontsize=14, fontweight="bold")
ax[0].set_ylim(0.0, 1.0)
ax[0].set_yticks([0.0, 0.5, 1.0])

ax[1].bar(r_ct_pred.keys(), r_ct_pred.values(), color="blue")
ax[1].set_ylabel("Multivariate Cortical Thickness (R)")
ax[1].set_title("B", loc="left", fontsize=14, fontweight="bold")
ax[1].set_ylim(0.0, 1.0)
ax[1].set_yticks([0.0, 0.5, 1.0])

ax[2] = sns.boxplot(data=test_r_sa, showfliers=False, ax=ax[2], color="blue")
ax[2].set_ylabel("Test Surface Area CV (R)")
ax[2].set_title("C", loc="left", fontsize=14, fontweight="bold")
ax[2].set_ylim(-1.0, 1.0)
ax[2].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

ax[3] = sns.boxplot(data=test_r_ct, showfliers=False, ax=ax[3], color="blue")
ax[3].set_ylabel("Test Cortical Thickness CV (R)")
ax[3].set_title("D", loc="left", fontsize=14, fontweight="bold")
ax[3].set_ylim(-1.0, 1.0)
ax[3].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax[3].set_xticks(np.arange(len(ct_disorders)), labels=ct_disorders)
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[2].spines["top"].set_visible(False)
ax[2].spines["right"].set_visible(False)
ax[3].spines["top"].set_visible(False)
ax[3].spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("diffusion_neuromaps/plots/figs/disorder_cv.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

###  4. Inspect CV test correlations region-wise
economo_pal = sns.color_palette("tab10", 5)
mesulam_pal = sns.color_palette("tab10")[5:9]
economo_regions = ["agranular", "frontal", "parietal", "polar", "granular"]
mesulam_regions = ["paralimbic", "heteromodal", "unimodal", "idiotypic"]

dk_parc = pd.read_csv(os.path.join("diffusion_neuromaps", "atlases", "dk_parc.csv"), index_col=0)
df_economo = pd.DataFrame(columns=["Region", "Disorder", "Value"])
df_mesulam = pd.DataFrame(columns=["Region", "Disorder", "Value"])
for name in ct_disorders:
    ct_r = test_r_ct[name]
    plot_dk_map(pd.DataFrame(ct_r, index=dk_regions, columns=[name]), size=(1600, 400), cmap="RdBu_r", label_text=None, color_bar=False, color_range="sym",
                    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
                    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "dk", f"{name}_ct_cv.png"))
    df_economo = pd.concat([df_economo, pd.DataFrame({"Region": dk_parc["economo"], "Disorder": name, "Value": ct_r})], axis=0)
    df_mesulam = pd.concat([df_mesulam, pd.DataFrame({"Region": dk_parc["mesulam"], "Disorder": name, "Value": ct_r})], axis=0)

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey="all")
ax[0] = sns.boxplot(data=df_economo, x="Disorder", y="Value", hue="Region", showfliers=False, ax=ax[0], order=ct_disorders, hue_order=economo_regions, palette=economo_pal)
ax[0].legend(title="", loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1))
ax[0].set_ylabel("Cross Validation (R)")
ax[0].set_ylim(-0.5, 1.0)
ax[0].set_yticks([-0.5, 0.0, 0.5, 1.0])
ax[0].set_xlabel("")

ax[1] = sns.boxplot(data=df_mesulam, x="Disorder", y="Value", hue="Region", showfliers=False, ax=ax[1], order=ct_disorders, hue_order=mesulam_regions, palette=mesulam_pal)
ax[1].legend(title="", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1))
ax[1].set_ylim(-0.5, 1.0)
ax[1].set_yticks([-0.5, 0.0, 0.5, 1.0])
ax[1].set_ylabel("")
ax[1].set_xlabel("")

ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("diffusion_neuromaps/plots/figs/disorder_cv_regionwise.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)


