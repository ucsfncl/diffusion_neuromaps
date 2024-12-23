import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


r2_df = pd.read_csv("diffusion_neuromaps/data/ridge/r2_metric.csv")
r2_df["r2"] = numpy.clip(r2_df["r2"], 0, 1)
targets = ["CogFluidComp_Unadj", "CogTotalComp_Unadj", "CogCrystalComp_Unadj", "Age_in_Yrs"]
target_names = ["Fluid Composite", "Total Cognition", "Crystalized Composite", "Age"]

metrics = ["fa", "md", "ad", "rd", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
           "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]
dti_metrics = ["fa", "md", "ad", "rd"]
dki_dti_metrics = ["dki-fa", "dki-ad", "dki-rd", "dki-md"]
dki_metrics = ["mk", "ak", "rk", "kfa", "mkt"]
noddi_metrics = ["icvf", "odi", "isovf"]
mapmri_metrics = ["msd", "qiv", "rtop", "rtap", "rtpp"]

for tgt_idx, target in enumerate(targets):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax = sns.boxplot(x="type", y="r2", data=r2_df[r2_df["target"] == target], ax=ax, showfliers=False, fill=False,
    palette={"fa": "red", "md": "red", "ad": "red", "rd": "red", 
"dki-fa": "purple", "dki-ad": "purple", "dki-rd": "purple", "dki-md": "purple", 
"mk": "orange", "ak": "orange", "rk": "orange", "kfa": "orange", "mkt": "orange", 
"icvf": "blue", "odi": "blue", "isovf": "blue", 
"msd": "green", "qiv": "green", "rtop": "green", "rtap": "green", "rtpp": "green", 
"thick": "black", "myl": "black"}, order=metrics)
    for i, txt in enumerate(metrics):
        if txt in dti_metrics:
            ax.get_xticklabels()[i].set_color("red")
        elif txt in dki_dti_metrics:
            ax.get_xticklabels()[i].set_color("purple")
        elif txt in dki_metrics:
            ax.get_xticklabels()[i].set_color("orange")
        elif txt in noddi_metrics:
            ax.get_xticklabels()[i].set_color("blue")
        elif txt in mapmri_metrics:
            ax.get_xticklabels()[i].set_color("green")
        else:
            ax.get_xticklabels()[i].set_color("black")
    ax.set_xticklabels([x.upper() for x in metrics], rotation=90, fontsize=14)
    ax.set_ylabel('Coefficient of Determination', fontsize=14)
    ax.set_xlabel("")
    ax.set_title(target_names[tgt_idx], fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"diffusion_neuromaps/plots/figs/ridge_metric_{target}.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

r2_df = pd.read_csv("diffusion_neuromaps/data/ridge/r2_factor.csv")
r2_df["r2"] = numpy.clip(r2_df["r2"], 0, 1)
targets = ["CogFluidComp_Unadj", "CogTotalComp_Unadj", "CogCrystalComp_Unadj", "Age_in_Yrs"]
target_names = ["Fluid Composite", "Total Cognition", "Crystalized Composite", "Age"]

metrics = ["F1", "F2", "F3", "F4"]

for tgt_idx, target in enumerate(targets):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax = sns.boxplot(x="type", y="r2", data=r2_df[r2_df["target"] == target], ax=ax, showfliers=False, fill=False,
    palette="tab10", order=metrics)
    ax.set_xticklabels([x.upper() for x in metrics], rotation=90, fontsize=14)
    ax.set_ylabel('Coefficient of Determination', fontsize=14)
    ax.set_xlabel("")
    ax.set_title(target_names[tgt_idx], fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"diffusion_neuromaps/plots/figs/ridge_factor_{target}.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)