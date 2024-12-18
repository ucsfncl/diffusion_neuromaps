from tools import plot_glasser_map, plot_glasser_hemi
import os
import numpy as np
import pandas as pd


dmri_metrics = [
    "fa", "ad", "rd", "md", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
    "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]

for metric in dmri_metrics:
    ave = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "ave", f"{metric}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
    li = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "li", f"{metric}_glasser_li.csv"), index_col=0).values.ravel().astype(float)
    intersubject_cov = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "intersubject_cov", f"{metric}_glasser_intersubject_cov.csv"), index_col=0).values.ravel().astype(float)
    icc = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "icc", f"{metric}_glasser_icc.csv"), index_col=0).values.ravel().astype(float)
    retest_cov = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "retest_cov", f"{metric}_glasser_retest_cov.csv"), index_col=0).values.ravel().astype(float)

    ave = np.clip(ave, ave.mean() - 4 * ave.std(), ave.mean() + 4 * ave.std())
    li = np.clip(li, li.mean() - 4 * li.std(), li.mean() + 4 * li.std())
    intersubject_cov = np.clip(intersubject_cov, intersubject_cov.mean() - 4 * intersubject_cov.std(), intersubject_cov.mean() + 4 * intersubject_cov.std())

    plot_glasser_map(ave, size=(1600, 400), cmap="Reds", label_text=None, color_bar=False, 
    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False, 
    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", f"{metric}_glasser_ave.png"))

    plot_glasser_hemi(li, layout = ["lh", "lh"], view=["lateral", "medial"], size=(800, 400), 
    cmap="RdBu_r", label_text=None, color_bar=None, nan_color=(255, 255, 255, 1), share="row",
    color_range="sym", zoom=1.25, transparent_bg=False, interactive=False, 
    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", f"{metric}_glasser_li.png"))

    plot_glasser_map(intersubject_cov, size=(1600, 400), cmap="Reds", label_text=None, color_bar=False,
    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", f"{metric}_glasser_intersubject_cov.png"))

    plot_glasser_map(icc, size=(1800, 400), cmap="Reds", label_text=None, color_bar=True,
    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", f"{metric}_glasser_icc.png"))

    plot_glasser_map(retest_cov, size=(1800, 400), cmap="Reds", label_text=None, color_bar=True,
    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", f"{metric}_glasser_retest_cov.png"))


