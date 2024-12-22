from tools import plot_glasser_map, plot_glasser_hemi
import os
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from tools import fdr_correction
import seaborn as sns
import matplotlib.pyplot as plt

metrics = [
    "fa", "ad", "rd", "md", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
    "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]

df_parc = pd.read_csv(os.path.join("diffusion_neuromaps", "atlases", "glasser_parc.csv"), index_col=0)

df_economo = pd.DataFrame(columns=["metric", "region", "value"])
df_mesulam = pd.DataFrame(columns=["metric", "region", "value"])
df_yeo = pd.DataFrame(columns=["metric", "region", "value"])
df_sa = pd.DataFrame(columns=["metric", "region", "value"])

df_economo_li = pd.DataFrame(columns=["metric", "region", "value"])
df_mesulam_li = pd.DataFrame(columns=["metric", "region", "value"])
df_yeo_li = pd.DataFrame(columns=["metric", "region", "value"])
df_sa_li = pd.DataFrame(columns=["metric", "region", "value"])

df_economo_cov = pd.DataFrame(columns=["metric", "region", "value"])
df_mesulam_cov = pd.DataFrame(columns=["metric", "region", "value"])
df_yeo_cov = pd.DataFrame(columns=["metric", "region", "value"])
df_sa_cov = pd.DataFrame(columns=["metric", "region", "value"])

economo_regions = ["agranular", "frontal", "parietal", "polar", "granular"]
mesulam_regions = ["paralimbic", "heteromodal", "unimodal", "idiotypic"]
sa_axis_regions = ["sensorimotor", "middle", "association"]
yeo_regions = ["visual", "somatosensory", "dorsal attention", "ventral attention", "limbic", "frontoparietal", "default mode"]

p_sa = {}
p_sa_li = {}
p_sa_cov = {}

p_economo = {}
p_economo_li = {}
p_economo_cov = {}

p_mesulam = {}
p_mesulam_li = {}
p_mesulam_cov = {}

p_yeo = {}
p_yeo_li = {}
p_yeo_cov = {}

T_sa = {}
T_sa_li = {}
T_sa_cov = {}

F_economo = {}
F_economo_li = {}
F_economo_cov = {}

F_mesulam = {}
F_mesulam_li = {}
F_mesulam_cov = {}

F_yeo = {}
F_yeo_li = {}
F_yeo_cov = {}

for metric in metrics:
    ave = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "ave", f"{metric}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
    li = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "li", f"{metric}_glasser_li.csv"), index_col=0).values.ravel().astype(float)
    intersubject_cov = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "intersubject_cov", f"{metric}_glasser_intersubject_cov.csv"), index_col=0).values.ravel().astype(float)

    ave = (ave - np.mean(ave)) / np.std(ave)
    li = li / np.std(li)
    intersubject_cov = (intersubject_cov - np.mean(intersubject_cov)) / np.std(intersubject_cov)

    res = scipy.stats.ttest_ind(ave[df_parc["sa"] == "association"], ave[df_parc["sa"] == "sensorimotor"], equal_var=False)
    T_sa[metric] = res.statistic
    p_sa[metric] = res.pvalue
    res = scipy.stats.ttest_ind(li[df_parc["sa"][:180] == "association"], li[df_parc["sa"][:180] == "sensorimotor"], equal_var=False)
    T_sa_li[metric] = res.statistic
    p_sa_li[metric] = res.pvalue
    res = scipy.stats.ttest_ind(intersubject_cov[df_parc["sa"] == "association"], intersubject_cov[df_parc["sa"] == "sensorimotor"], equal_var=False)
    T_sa_cov[metric] = res.statistic
    p_sa_cov[metric] = res.pvalue

    F_economo[metric], p_economo[metric] = scipy.stats.f_oneway(*[ave[df_parc["economo"] == region] for region in economo_regions])
    F_economo_li[metric], p_economo_li[metric] = scipy.stats.f_oneway(*[li[df_parc["economo"][:180] == region] for region in economo_regions])
    F_economo_cov[metric], p_economo_cov[metric] = scipy.stats.f_oneway(*[intersubject_cov[df_parc["economo"] == region] for region in economo_regions])

    F_mesulam[metric], p_mesulam[metric] = scipy.stats.f_oneway(*[ave[df_parc["mesulam"] == region] for region in mesulam_regions])
    F_mesulam_li[metric], p_mesulam_li[metric] = scipy.stats.f_oneway(*[li[df_parc["mesulam"][:180] == region] for region in mesulam_regions])
    F_mesulam_cov[metric], p_mesulam_cov[metric] = scipy.stats.f_oneway(*[intersubject_cov[df_parc["mesulam"] == region] for region in mesulam_regions])

    F_yeo[metric], p_yeo[metric] = scipy.stats.f_oneway(*[ave[df_parc["yeo"] == region] for region in yeo_regions])
    F_yeo_li[metric], p_yeo_li[metric] = scipy.stats.f_oneway(*[li[df_parc["yeo"][:180] == region] for region in yeo_regions])
    F_yeo_cov[metric], p_yeo_cov[metric] = scipy.stats.f_oneway(*[intersubject_cov[df_parc["yeo"] == region] for region in yeo_regions])

    df_sa = pd.concat([df_sa, pd.DataFrame({"metric": metric, "region": df_parc["sa"], "value": ave})], ignore_index=True)
    df_sa_li = pd.concat([df_sa_li, pd.DataFrame({"metric": metric, "region": df_parc["sa"][:180], "value": li})], ignore_index=True)
    df_sa_cov = pd.concat([df_sa_cov, pd.DataFrame({"metric": metric, "region": df_parc["sa"], "value": intersubject_cov})], ignore_index=True)

    df_economo = pd.concat([df_economo, pd.DataFrame({"metric": metric, "region": df_parc["economo"], "value": ave})], ignore_index=True)
    df_economo_li = pd.concat([df_economo_li, pd.DataFrame({"metric": metric, "region": df_parc["economo"][:180], "value": li})], ignore_index=True)
    df_economo_cov = pd.concat([df_economo_cov, pd.DataFrame({"metric": metric, "region": df_parc["economo"], "value": intersubject_cov})], ignore_index=True)

    df_mesulam = pd.concat([df_mesulam, pd.DataFrame({"metric": metric, "region": df_parc["mesulam"], "value": ave})], ignore_index=True)
    df_mesulam_li = pd.concat([df_mesulam_li, pd.DataFrame({"metric": metric, "region": df_parc["mesulam"][:180], "value": li})], ignore_index=True)
    df_mesulam_cov = pd.concat([df_mesulam_cov, pd.DataFrame({"metric": metric, "region": df_parc["mesulam"], "value": intersubject_cov})], ignore_index=True)

    df_yeo = pd.concat([df_yeo, pd.DataFrame({"metric": metric, "region": df_parc["yeo"], "value": ave})], ignore_index=True)
    df_yeo_li = pd.concat([df_yeo_li, pd.DataFrame({"metric": metric, "region": df_parc["yeo"][:180], "value": li})], ignore_index=True)
    df_yeo_cov = pd.concat([df_yeo_cov, pd.DataFrame({"metric": metric, "region": df_parc["yeo"], "value": intersubject_cov})], ignore_index=True)

p_sa = fdr_correction(p_sa)
p_sa_li = fdr_correction(p_sa_li)
p_sa_cov = fdr_correction(p_sa_cov)

p_economo = fdr_correction(p_economo)
p_economo_li = fdr_correction(p_economo_li)
p_economo_cov = fdr_correction(p_economo_cov)

p_mesulam = fdr_correction(p_mesulam)
p_mesulam_li = fdr_correction(p_mesulam_li)
p_mesulam_cov = fdr_correction(p_mesulam_cov)

p_yeo = fdr_correction(p_yeo)
p_yeo_li = fdr_correction(p_yeo_li)
p_yeo_cov = fdr_correction(p_yeo_cov)

print(p_sa)
print(p_economo)
print(p_mesulam)
print(p_yeo)

yeo_pal = sns.color_palette("tab10", 7)
economo_pal = sns.color_palette("tab10", 5)
mesulam_pal = sns.color_palette("tab10")[5:9]
sa_pal = {"association": "red", "sensorimotor": "blue"}

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(3, 1, figsize=(16, 18), gridspec_kw={"hspace": 0.1, "wspace": 0.0}, sharex=True)
for i in range(len(metrics) - 1):
    ax[0].axvline(i + 0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax[0].axhline(0.0, color="blue", linewidth=1.0, alpha=1.0)
ax[0] = sns.boxplot(ax=ax[0], data=df_yeo, x="metric", y="value", hue="region", order=metrics, showfliers=False, hue_order=yeo_regions, palette=yeo_pal)
ax[0].legend(loc='upper center', ncol=5, title="", fontsize=14, bbox_to_anchor=(0.5, 1.05))
ax[0].set_yticks([3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
ax[0].set_ylabel("Metric Z-Score", fontsize=14)
ax[0].set_xlabel("")
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_ylim(-3.5, 4.0)

for i in range(len(metrics) - 1):
    ax[1].axvline(i + 0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax[1].axhline(0.0, color="blue", linewidth=1.0, alpha=1.0)

ax[1] = sns.boxplot(ax=ax[1], data=df_yeo_cov, x="metric", y="value", hue="region", order=metrics, showfliers=False, hue_order=yeo_regions, palette=yeo_pal, legend=False)
ax[1].set_yticks([ -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
ax[1].set_ylabel("Intersubject CoV Z-Score", fontsize=14)
ax[1].set_xlabel("")
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_ylim(-3.0, 4.0)

for i in range(len(metrics) - 1):
    ax[2].axvline(i + 0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax[2].axhline(0.0, color="blue", linewidth=1.0, alpha=1.0)
ax[2] = sns.boxplot(ax=ax[2], data=df_yeo_li, x="metric", y="value", hue="region", order=metrics, showfliers=False, hue_order=yeo_regions, palette=yeo_pal, legend=False)
ax[2].set_xticks(np.arange(len(metrics)), labels=[metric.upper() for metric in metrics], rotation=270, fontsize=14)
ax[2].set_yticks([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
ax[2].set_ylabel("LI Z-Score", fontsize=14)
ax[2].set_xlabel("")
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
ax[2].set_ylim(-3.5, 3.0)
for i in range(4):
    ax[2].get_xticklabels()[i].set_color("red")
for i in range(4, 8):
    ax[2].get_xticklabels()[i].set_color("purple")
for i in range(8, 13):
    ax[2].get_xticklabels()[i].set_color("orange")
for i in range(13, 16):
    ax[2].get_xticklabels()[i].set_color("blue")
for i in range(16, 21):
    ax[2].get_xticklabels()[i].set_color("green")

fig.tight_layout()
fig.savefig(os.path.join("diffusion_neuromaps", "plots", "figs", "fig_yeo.png"), bbox_inches="tight", dpi=400, pad_inches=0.05)
plt.close(fig)

fig, ax = plt.subplots(3, 1, figsize=(16, 18), gridspec_kw={"hspace": 0.1, "wspace": 0.0}, sharex=True)
for i in range(len(metrics) - 1):
    ax[0].axvline(i + 0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax[0].axhline(0.0, color="blue", linewidth=1.0, alpha=1.0)
ax[0] = sns.boxplot(ax=ax[0], data=df_economo, x="metric", y="value", hue="region", order=metrics, showfliers=False, hue_order=economo_regions, palette=economo_pal)
ax[0].legend(loc='upper center', ncol=5, title="", fontsize=14, bbox_to_anchor=(0.5, 1.05))
ax[0].set_yticks([3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
ax[0].set_ylabel("Metric Z-Score", fontsize=14)
ax[0].set_xlabel("")
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_ylim(-3.5, 4.0)

for i in range(len(metrics) - 1):
    ax[1].axvline(i + 0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax[1].axhline(0.0, color="blue", linewidth=1.0, alpha=1.0)

ax[1] = sns.boxplot(ax=ax[1], data=df_economo_cov, x="metric", y="value", hue="region", order=metrics, showfliers=False, hue_order=economo_regions, palette=economo_pal, legend=False)
ax[1].set_yticks([ -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
ax[1].set_ylabel("Intersubject CoV Z-Score", fontsize=14)
ax[1].set_xlabel("")
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_ylim(-3.0, 4.0)

for i in range(len(metrics) - 1):
    ax[2].axvline(i + 0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax[2].axhline(0.0, color="blue", linewidth=1.0, alpha=1.0)
ax[2] = sns.boxplot(ax=ax[2], data=df_economo_li, x="metric", y="value", hue="region", order=metrics, showfliers=False, hue_order=economo_regions, palette=economo_pal, legend=False)
ax[2].set_xticks(np.arange(len(metrics)), labels=[metric.upper() for metric in metrics], rotation=270, fontsize=14)
ax[2].set_yticks([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
ax[2].set_ylabel("LI Z-Score", fontsize=14)
ax[2].set_xlabel("")
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
for i in range(4):
    ax[2].get_xticklabels()[i].set_color("red")
for i in range(4, 8):
    ax[2].get_xticklabels()[i].set_color("purple")
for i in range(8, 13):
    ax[2].get_xticklabels()[i].set_color("orange")
for i in range(13, 16):
    ax[2].get_xticklabels()[i].set_color("blue")
for i in range(16, 21):
    ax[2].get_xticklabels()[i].set_color("green")

fig.tight_layout()
fig.savefig(os.path.join("diffusion_neuromaps", "plots", "figs", "fig_economo.png"), bbox_inches="tight", dpi=400, pad_inches=0.05)


fig, ax = plt.subplots(3, 1, figsize=(16, 18), gridspec_kw={"hspace": 0.1, "wspace": 0.0}, sharex=True)
for i in range(len(metrics) - 1):
    ax[0].axvline(i + 0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax[0].axhline(0.0, color="blue", linewidth=1.0, alpha=1.0)
ax[0] = sns.boxplot(ax=ax[0], data=df_mesulam, x="metric", y="value", hue="region", order=metrics, showfliers=False, hue_order=mesulam_regions, palette=mesulam_pal)
ax[0].legend(loc='upper center', ncol=5, title="", fontsize=14, bbox_to_anchor=(0.5, 1.05))
ax[0].set_yticks([3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
ax[0].set_ylabel("Metric Z-Score", fontsize=14)
ax[0].set_xlabel("")
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_ylim(-3.5, 4.0)

for i in range(len(metrics) - 1):
    ax[1].axvline(i + 0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax[1].axhline(0.0, color="blue", linewidth=1.0, alpha=1.0)

ax[1] = sns.boxplot(ax=ax[1], data=df_mesulam_cov, x="metric", y="value", hue="region", order=metrics, showfliers=False, hue_order=mesulam_regions, palette=mesulam_pal, legend=False)
ax[1].set_yticks([ -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
ax[1].set_ylabel("Intersubject CoV Z-Score", fontsize=14)
ax[1].set_xlabel("")
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_ylim(-3.0, 4.0)

for i in range(len(metrics) - 1):
    ax[2].axvline(i + 0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax[2].axhline(0.0, color="blue", linewidth=1.0, alpha=1.0)
ax[2] = sns.boxplot(ax=ax[2], data=df_mesulam_li, x="metric", y="value", hue="region", order=metrics, showfliers=False, hue_order=mesulam_regions, palette=mesulam_pal, legend=False)
ax[2].set_xticks(np.arange(len(metrics)), labels=[metric.upper() for metric in metrics], rotation=270, fontsize=14)
ax[2].set_yticks([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
ax[2].set_ylabel("LI Z-Score", fontsize=14)
ax[2].set_xlabel("")
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
for i in range(4):
    ax[2].get_xticklabels()[i].set_color("red")
for i in range(4, 8):
    ax[2].get_xticklabels()[i].set_color("purple")
for i in range(8, 13):
    ax[2].get_xticklabels()[i].set_color("orange")
for i in range(13, 16):
    ax[2].get_xticklabels()[i].set_color("blue")
for i in range(16, 21):
    ax[2].get_xticklabels()[i].set_color("green")

fig.tight_layout()
fig.savefig(os.path.join("diffusion_neuromaps", "plots", "figs", "fig_mesulam.png"), bbox_inches="tight", dpi=400, pad_inches=0.05)


fig, ax = plt.subplots(3, 1, figsize=(16, 18), gridspec_kw={"hspace": 0.1, "wspace": 0.0}, sharex=True)
for i in range(len(metrics) - 1):
    ax[0].axvline(i + 0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax[0].axhline(0.0, color="blue", linewidth=1.0, alpha=1.0)
ax[0] = sns.boxplot(ax=ax[0], data=df_sa, x="metric", y="value", hue="region", order=metrics, showfliers=False, hue_order=["association", "sensorimotor"], palette=sa_pal)
ax[0].legend(loc='upper center', ncol=2, title="", fontsize=14, bbox_to_anchor=(0.5, 1.05))
ax[0].set_yticks([3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
ax[0].set_ylabel("Metric Z-Score", fontsize=14)
ax[0].set_xlabel("")
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_ylim(-3.5, 4.0)

for i in range(len(metrics) - 1):
    ax[1].axvline(i + 0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax[1].axhline(0.0, color="blue", linewidth=1.0, alpha=1.0)

ax[1] = sns.boxplot(ax=ax[1], data=df_sa_cov, x="metric", y="value", hue="region", order=metrics, showfliers=False, hue_order=["association", "sensorimotor"], palette=sa_pal, legend=False)
ax[1].set_yticks([ -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
ax[1].set_ylabel("Intersubject CoV Z-Score", fontsize=14)
ax[1].set_xlabel("")
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_ylim(-3.0, 4.0)

for i in range(len(metrics) - 1):
    ax[2].axvline(i + 0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax[2].axhline(0.0, color="blue", linewidth=1.0, alpha=1.0)
ax[2] = sns.boxplot(ax=ax[2], data=df_sa_li, x="metric", y="value", hue="region", order=metrics, showfliers=False, hue_order=["association", "sensorimotor"], palette=sa_pal, legend=False)
ax[2].set_xticks(np.arange(len(metrics)), labels=[metric.upper() for metric in metrics], rotation=270, fontsize=14)
ax[2].set_yticks([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
ax[2].set_ylabel("LI Z-Score", fontsize=14)
ax[2].set_xlabel("")
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
for i in range(4):
    ax[2].get_xticklabels()[i].set_color("red")
for i in range(4, 8):
    ax[2].get_xticklabels()[i].set_color("purple")
for i in range(8, 13):
    ax[2].get_xticklabels()[i].set_color("orange")
for i in range(13, 16):
    ax[2].get_xticklabels()[i].set_color("blue")
for i in range(16, 21):
    ax[2].get_xticklabels()[i].set_color("green")

fig.tight_layout()
fig.savefig(os.path.join("diffusion_neuromaps", "plots", "figs", "fig_sa.png"), bbox_inches="tight", dpi=400, pad_inches=0.05)
plt.close(fig)