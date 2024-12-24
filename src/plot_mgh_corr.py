import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


###  1. Load the microstructural glasser maps from HCP and MGH
dti_metrics = ["fa", "ad", "rd", "md"]
dki_dti_metrics = ["dki-fa", "dki-ad", "dki-rd", "dki-md"]
dki_metrics = ["mk", "ak", "rk", "kfa", "mkt"]
noddi_metrics = ["icvf", "odi", "isovf"]
mapmri_metrics = ["msd", "qiv", "rtop", "rtap", "rtpp"]
struct_metrics = ["thick", "myl"]

metrics = [
    "fa", "ad", "rd", "md", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
    "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]
glasser_df = pd.read_csv(os.path.join("diffusion_neuromaps", "atlases", "glasser_parc.csv"), index_col=0)
mgh_df = pd.DataFrame(index=glasser_df.index, columns=metrics)
for metric in metrics:
    ave = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "mgh", f"{metric}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
    mgh_df[metric] = ave

hcp_df = pd.DataFrame(index=glasser_df.index, columns=metrics)
for metric in metrics:
    ave = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "ave", f"{metric}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
    hcp_df[metric] = ave

limited_metrics = ["dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
                        "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp"]
mgh_limited_df = pd.DataFrame(index=glasser_df.index, columns=limited_metrics)
for metric in limited_metrics:
    ave = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "mgh", f"{metric}-limited_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
    mgh_limited_df[metric] = ave


###  2. Plot the Pearson correlation between HCP and MGH
r = {key: np.corrcoef(hcp_df[key], mgh_df[key])[0, 1] for key in metrics}
r_limited = {key: np.corrcoef(hcp_df[key], mgh_limited_df[key])[0, 1] for key in limited_metrics}

fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True, width_ratios=[23, 17])
ax[0] = sns.barplot(x=metrics, y=[r[key] for key in metrics], alpha=0.5, order=metrics, ax=ax[0])
ax[0].set_ylabel("Pearson Correlation (r)", fontsize=14)
ax[0].set_title("HCP vs. MGH", fontsize=14)
ax[0].set_xlabel("")
ax[0].set_ylim(0.85, 1.0)
ax[0].set_xticklabels(metrics, rotation=90, fontsize=14)

for i, key in enumerate(metrics):
    if key in dti_metrics:
        ax[0].get_children()[i].set_facecolor("red")
        ax[0].get_children()[i].set_edgecolor("black")
        ax[0].get_xticklabels()[i].set_color("red")
    elif key in dki_dti_metrics:
        ax[0].get_children()[i].set_facecolor("purple")
        ax[0].get_children()[i].set_edgecolor("black")
        ax[0].get_xticklabels()[i].set_color("purple")
    elif key in dki_metrics:
        ax[0].get_children()[i].set_facecolor("orange")
        ax[0].get_children()[i].set_edgecolor("black")
        ax[0].get_xticklabels()[i].set_color("orange")
    elif key in noddi_metrics:
        ax[0].get_children()[i].set_facecolor("blue")
        ax[0].get_children()[i].set_edgecolor("black")
        ax[0].get_xticklabels()[i].set_color("blue")
    elif key in mapmri_metrics:
        ax[0].get_children()[i].set_facecolor("green")
        ax[0].get_children()[i].set_edgecolor("black")
        ax[0].get_xticklabels()[i].set_color("green")
    elif key in struct_metrics:
        ax[0].get_children()[i].set_facecolor("black")
        ax[0].get_children()[i].set_edgecolor("black")
        ax[0].get_xticklabels()[i].set_color("black")
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)


ax[1].set_title("HCP vs. MGH (Limited)", fontsize=14)
ax[1] = sns.barplot(x=limited_metrics, y=[r_limited[key] for key in limited_metrics], alpha=0.5, order=limited_metrics, ax=ax[1])
ax[1].set_ylabel("Pearson Correlation (r)", fontsize=14)
ax[1].set_xlabel("")
ax[1].set_ylim(0.85, 1.0)
ax[1].set_xticklabels(limited_metrics, rotation=90, fontsize=14)

for i, key in enumerate(limited_metrics):
    if key in dki_dti_metrics:
        ax[1].get_children()[i].set_facecolor("purple")
        ax[1].get_children()[i].set_edgecolor("black")
        ax[1].get_xticklabels()[i].set_color("purple")
    elif key in dki_metrics:
        ax[1].get_children()[i].set_facecolor("orange")
        ax[1].get_children()[i].set_edgecolor("black")
        ax[1].get_xticklabels()[i].set_color("orange")
    elif key in noddi_metrics:
        ax[1].get_children()[i].set_facecolor("blue")
        ax[1].get_children()[i].set_edgecolor("black")
        ax[1].get_xticklabels()[i].set_color("blue")
    elif key in mapmri_metrics:
        ax[1].get_children()[i].set_facecolor("green")
        ax[1].get_children()[i].set_edgecolor("black")
        ax[1].get_xticklabels()[i].set_color("green")

ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig("diffusion_neuromaps/plots/figs/mgh_hcp_ave_corr.png", bbox_inches="tight", dpi=400, pad_inches=0.05)
plt.close()


###  3. Plot the scatter plot of the correlation between HCP and MGH

for metric in limited_metrics:

    fontcolor = "black"
    if metric in dki_dti_metrics:
        fontcolor = "purple"
    elif metric in dki_metrics:
        fontcolor = "orange"
    elif metric in noddi_metrics:
        fontcolor = "blue"
    elif metric in mapmri_metrics:
        fontcolor = "green"
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax = sns.regplot(x=hcp_df[metric], y=mgh_df[metric], ax=ax, scatter_kws={"s": 10, "alpha": 0.5}, color="blue", label="Full")
    ax = sns.regplot(x=hcp_df[metric], y=mgh_limited_df[metric], ax=ax, scatter_kws={"s": 10, "alpha": 0.5}, color="red", label="Limited")
    ax.plot([np.min(hcp_df[metric]), np.max(hcp_df[metric])], [np.min(hcp_df[metric]), np.max(hcp_df[metric])], color="black", linestyle="--")
    ax.legend(loc="upper left")
    ax.set_xlabel(f"HCP {metric.upper()}", color=fontcolor)
    ax.set_ylabel(f"MGH {metric.upper()}", color=fontcolor)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig.tight_layout()
    fig.savefig(f"diffusion_neuromaps/plots/mgh/mgh-{metric}-scatter.png", bbox_inches="tight", dpi=400, pad_inches=0.05)
    plt.close(fig)

metrics = ["fa", "ad", "rd", "md", "thick", "myl"]
for metric in metrics:
    
    fontcolor = "black"
    if metric in dti_metrics:
        fontcolor = "red"
    elif metric in struct_metrics:
        fontcolor = "black"

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax = sns.regplot(x=hcp_df[metric], y=mgh_df[metric], ax=ax, scatter_kws={"s": 10, "alpha": 0.5}, color="blue", label="Full")
    ax.plot([np.min(hcp_df[metric]), np.max(hcp_df[metric])], [np.min(hcp_df[metric]), np.max(hcp_df[metric])], color="black", linestyle="--")
    ax.legend(loc="upper left")
    ax.set_xlabel(f"HCP {metric.upper()}", color=fontcolor)
    ax.set_ylabel(f"MGH {metric.upper()}", color=fontcolor)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig.tight_layout()
    fig.savefig(f"diffusion_neuromaps/plots/mgh/mgh-{metric}-scatter.png", bbox_inches="tight", dpi=400, pad_inches=0.05)
    plt.close(fig) 
