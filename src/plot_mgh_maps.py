from tools import plot_glasser_map
import os
import numpy as np
import pandas as pd


dmri_metrics = [
    "fa", "ad", "rd", "md", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
    "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl", "F1", "F2", "F3", "F4"]

for metric in dmri_metrics:
    ave = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "mgh", f"{metric}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)

    plot_glasser_map(ave, size=(1600, 400), cmap="Reds", label_text=None, color_bar=False, 
    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False, 
    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "mgh", f"mgh_{metric}_glasser_ave.png"))


dmri_limited_metrics = ["dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
                        "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "F1", "F2", "F3", "F4"]

for metric in dmri_limited_metrics:
    ave = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "mgh", f"{metric}-limited_glasser_ave.csv"), index_col=0).values.ravel().astype(float)

    plot_glasser_map(ave, size=(1600, 400), cmap="Reds", label_text=None, color_bar=False, 
    nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False, 
    screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "mgh", f"mgh_{metric}-limited_glasser_ave.png"))