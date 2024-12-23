import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import matplotlib.patches as patches
from tools import plot_glasser_map


metrics = ["fa", "md", "ad", "rd", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
           "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]
dti_metrics = ["fa", "md", "ad", "rd"]
dki_dti_metrics = ["dki-fa", "dki-ad", "dki-rd", "dki-md"]
dki_metrics = ["mk", "ak", "rk", "kfa", "mkt"]
noddi_metrics = ["icvf", "odi", "isovf"]
mapmri_metrics = ["msd", "qiv", "rtop", "rtap", "rtpp"]

factors = ["F1", "F2", "F3", "F4"]

glasser_df = pd.read_csv("diffusion_neuromaps/atlases/glasser_parc.csv", index_col=0)

folders = ["univariate", "cogtotal", "cogfluid", "cogcrystal"]
folder_names = ["$h^2$", "Total Cognition $h^2$", "Fluid Cognition $h^2$", "Crystal Cognition $h^2$"]


for idx, folder in enumerate(folders):
    hcp_df = pd.DataFrame(index=metrics, columns=glasser_df.index.to_list(), dtype=float)
    for metric in metrics:
        df = pd.read_csv(f"diffusion_neuromaps/data/heritability/{folder}/{metric}.csv", index_col=0)
        hcp_df.loc[metric] = df.values.ravel().astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.boxplot(data=hcp_df.T, ax=ax, showfliers=False, fill=False, 
    palette={"fa": "red", "md": "red", "ad": "red", "rd": "red", 
    "dki-fa": "purple", "dki-ad": "purple", "dki-rd": "purple", "dki-md": "purple", 
    "mk": "orange", "ak": "orange", "rk": "orange", "kfa": "orange", "mkt": "orange", 
    "icvf": "blue", "odi": "blue", "isovf": "blue", 
    "msd": "green", "qiv": "green", "rtop": "green", "rtap": "green", "rtpp": "green", 
    "thick": "black", "myl": "black"})
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
    ax.set_ylabel(folder_names[idx], fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"diffusion_neuromaps/plots/figs/{folder}_heritability.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    hcp_df = pd.DataFrame(index=factors, columns=glasser_df.index.to_list(), dtype=float)
    for factor in factors:
        df = pd.read_csv(f"diffusion_neuromaps/data/heritability/{folder}/{factor}.csv", index_col=0)
        hcp_df.loc[factor] = df.values.ravel().astype(float)
    
    fig, ax = plt.subplots(figsize=(4, 5))
    ax = sns.boxplot(data=hcp_df.T, ax=ax, showfliers=False, fill=False, palette="tab10")
    ax.set_xticklabels(factors, rotation=90, fontsize=14)
    ax.set_ylabel(folder_names[idx], fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"diffusion_neuromaps/plots/figs/{folder}_heritability_factors.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
