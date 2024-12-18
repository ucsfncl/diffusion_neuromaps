from typing import Dict
import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import curve_fit
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from nilearn.surface import load_surf_data
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_elements import get_points 
from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.null_models import SpinPermutations, MoranRandomization
from dominance_analysis import Dominance


def plot_glasser_map(arr: np.ndarray, **kwargs):
    """
    Plot a Glasser map with the given array values.
    """

    lh_glasser = load_surf_data("diffusion_neuromaps/atlases/Glasser_2016.32k.L.label.gii")
    rh_glaser = load_surf_data("diffusion_neuromaps/atlases/Glasser_2016.32k.R.label.gii")
    lh_stat = np.zeros(lh_glasser.shape)
    rh_stat = np.zeros(rh_glaser.shape)
    lh_stat.fill(np.nan)
    rh_stat.fill(np.nan)
    lh_surf = read_surface(os.path.join("diffusion_neuromaps", "atlases", "fs_LR.32k.L.very_inflated.surf.gii"))
    rh_surf = read_surface(os.path.join("diffusion_neuromaps", "atlases", "fs_LR.32k.R.very_inflated.surf.gii"))

    # iterate over every row in the atlas dataframe
    for i in range(180):
        lh_stat[lh_glasser == i + 1] = arr[i]
        rh_stat[rh_glaser == i + 1] = arr[i + 180]
    plot_hemispheres(lh_surf, rh_surf, array_name=np.concatenate((lh_stat, rh_stat)), **kwargs)

def plot_glasser_hemi(arr: np.ndarray, **kwargs):
    """
    Plot a hemisphere with the given array values.
    """
    lh_glasser = load_surf_data(os.path.join("diffusion_neuromaps", "atlases", "Glasser_2016.32k.L.label.gii"))
    lh_stat = np.zeros(lh_glasser.shape)
    lh_stat.fill(np.nan)
    lh_surf = read_surface(os.path.join("diffusion_neuromaps", "atlases", "fs_LR.32k.L.very_inflated.surf.gii"))
    for i in range(180):
        lh_stat[lh_glasser == i + 1] = arr[i]

    array_name2 = []
    name = lh_surf.append_array(lh_stat, at='p')
    array_name2.append(name)
    array_name = np.asarray(array_name2)[:, None]
    print(array_name.shape)

    plot_surf({"lh": lh_surf}, array_name=array_name, **kwargs)

    # plot_surf({"lh": lh_surf}, layout = ["lh", "lh"], array_name=array_name, view=["lateral", "medial"], size=(800, 400), 
    #           cmap="RdBu_r", label_text=label_text, color_bar=None, nan_color=(255, 255, 255, 1), share="row",
    #           color_range="sym", zoom=1.25, transparent_bg=False, interactive=False, screenshot=True, filename=save_path)


def make_glasser_nulls(n_rep: int = 9999, random_state: int = 0) -> SpinPermutations:
    """
    Make spin permutations for the Glasser atlas.
    """
    pts = np.loadtxt("diffusion_neuromaps/atlases/glasser_pts.txt")
    lh_pts = pts[:180]
    rh_pts = pts[180:]
    gen = SpinPermutations(n_rep=n_rep, random_state=random_state)
    gen.fit(lh_pts, points_rh=rh_pts)
    return gen

def spin_test_glasser(gen: SpinPermutations, x: np.ndarray, y: np.ndarray, alt: str = "both"):
    """
    Performs a permuted spin test on the Glasser atlas (controls for spatial autocorrelation).
    """
    nan_mask = np.isnan(y) | np.isnan(x)
    r = np.corrcoef(x[~nan_mask], y[~nan_mask])[0, 1]
    y_perm = np.hstack(gen.randomize(y[:180], y[180:]))
    surrogate_r = np.zeros(y_perm.shape[0])
    for i in range(y_perm.shape[0]):
        nan_mask = np.isnan(y_perm[i, :]) | np.isnan(x)
        surrogate_r[i] = np.corrcoef(x[~nan_mask], y_perm[i, ~nan_mask])[0, 1]
    if alt == "lower":
        p = (np.sum(surrogate_r < r) + 1) / (y_perm.shape[0] + 1)
    elif alt == "greater":
        p = (np.sum(surrogate_r > r) + 1) / (y_perm.shape[0] + 1)
    else:
        less = (np.sum(surrogate_r < r) + 1) / (y_perm.shape[0] + 1)
        greater = (np.sum(surrogate_r > r) + 1) / (y_perm.shape[0] + 1)
        p = 2 * min(less, greater)
    return r, surrogate_r, p

