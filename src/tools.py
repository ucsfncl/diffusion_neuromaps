from typing import Dict, Tuple
import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import scipy
import scipy.stats

from nilearn.surface import load_surf_data
from brainspace.mesh.mesh_io import read_surface
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.null_models import SpinPermutations
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

def plot_dk_map(df: pd.DataFrame, **kwargs):
    """
    Plot a Desikan-Killiany map with the given array values.
    """
    lh_dkt = load_surf_data("diffusion_neuromaps/atlases/Desikan.32k.L.label.gii")
    rh_dkt = load_surf_data("diffusion_neuromaps/atlases/Desikan.32k.R.label.gii")
    lh_stat = np.zeros(lh_dkt.shape)
    rh_stat = np.zeros(rh_dkt.shape)
    lh_stat.fill(np.nan)
    rh_stat.fill(np.nan)

    lh_surf = read_surface(os.path.join("diffusion_neuromaps", "atlases", "fs_LR.32k.L.very_inflated.surf.gii"))
    rh_surf = read_surface(os.path.join("diffusion_neuromaps", "atlases", "fs_LR.32k.R.very_inflated.surf.gii"))

    with open("diffusion_neuromaps/atlases/fslr_dk.json", "r") as f:
        fslr_labels = json.load(f)
    fslr_labels = {int(k): v for k, v in fslr_labels.items()}
    dkt_regions = [f"lh.{val}" for val in fslr_labels.values()] + [f"rh.{val}" for val in fslr_labels.values()]

    for key, val in fslr_labels.items():
        lh_stat[lh_dkt == key] = df.loc[f"lh.{val}"]
        rh_stat[rh_dkt == key] = df.loc[f"rh.{val}"]
    
    plot_hemispheres(lh_surf, rh_surf, array_name=np.concatenate((lh_stat, rh_stat)), **kwargs)

def make_glasser_nulls(n_rep: int = 9999, random_state: int = 0) -> SpinPermutations:
    """
    Make spin permutations for the Glasser atlas.
    """
    # pts = np.loadtxt("diffusion_neuromaps/atlases/glasser_spherical_pts.txt")
    pts = np.loadtxt("microstructure/atlases/glasser_pts.txt")
    lh_pts = pts[:180]
    rh_pts = pts[180:]
    gen = SpinPermutations(n_rep=n_rep, random_state=random_state)
    gen.fit(lh_pts, points_rh=rh_pts)
    return gen


def make_dk_nulls(n_rep: int = 9999, random_state: int = 0) -> SpinPermutations:
    """
    Make spin permutations for the Desikan-Killiany atlas.
    """
    pts = np.loadtxt("diffusion_neuromaps/atlases/dk_spherical_pts.txt")
    lh_pts = pts[:34]
    rh_pts = pts[34:]
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


def spin_test_dk(gen: SpinPermutations, x: np.ndarray, y: np.ndarray, alt: str = "both"):
    """
    Performs a permuted spin test on the Desikan-Killiany atlas (controls for spatial autocorrelation).
    """
    nan_mask = np.isnan(y) | np.isnan(x)
    r = np.corrcoef(x[~nan_mask], y[~nan_mask])[0, 1]
    y_perm = np.hstack(gen.randomize(y[:34], y[34:]))
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


def spin_test_adj_glasser(gen: SpinPermutations, x: np.ndarray, y: np.ndarray, alt: str = "both"):
    """
    Performs a permuted spin test on the Glasser atlas (controls for spatial autocorrelation).
    """
    mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
    r = np.corrcoef(x[mask], y[mask])[0, 1]
    x_mat = np.zeros((360, 360))
    triu_indices = np.triu_indices(360, k=1)
    tril_indices = np.tril_indices(360, k=-1)
    x_mat[triu_indices] = x
    x_mat[tril_indices] = x
    perm_indices = np.hstack(gen.randomize(np.arange(180), np.arange(180, 360)))
    surrogate_r = np.zeros(perm_indices.shape[0])   
    for i in range(perm_indices.shape[0]):
        surrogate_r[i] = np.corrcoef(x_mat[perm_indices[i], :][:, perm_indices[i]][triu_indices][mask], y[mask])[0, 1]
    if alt == "lower":
        p = (np.sum(surrogate_r <= r) + 1) / (perm_indices.shape[0] + 1)
    elif alt == "greater":
        p = (np.sum(surrogate_r >= r) + 1) / (perm_indices.shape[0] + 1)
    else:
        less = (np.sum(surrogate_r < r) + 1) / (perm_indices.shape[0] + 1)
        greater = (np.sum(surrogate_r > r) + 1) / (perm_indices.shape[0] + 1)
        p = 2 * min(less, greater)
    return r, surrogate_r, p


def spin_test_adj_dk(gen: SpinPermutations, x: np.ndarray, y: np.ndarray, alt: str = "both"):
    """
    Performs a permuted spin test on the Desikan-Killiany atlas (controls for spatial autocorrelation).
    """
    mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
    r = np.corrcoef(x[mask], y[mask])[0, 1]
    x_mat = np.zeros((68, 68))
    triu_indices = np.triu_indices(68, k=1)
    tril_indices = np.tril_indices(68, k=-1)
    x_mat[triu_indices] = x
    x_mat[tril_indices] = x
    perm_indices = np.hstack(gen.randomize(np.arange(34), np.arange(34, 68)))
    surrogate_r = np.zeros(perm_indices.shape[0])
    for i in range(perm_indices.shape[0]):
        surrogate_r[i] = np.corrcoef(x_mat[perm_indices[i], :][:, perm_indices[i]][triu_indices][mask], y[mask])[0, 1]
    if alt == "lower":
        p = (np.sum(surrogate_r <= r) + 1) / (perm_indices.shape[0] + 1)
    elif alt == "greater":
        p = (np.sum(surrogate_r >= r) + 1) / (perm_indices.shape[0] + 1)
    else:
        less = (np.sum(surrogate_r < r) + 1) / (perm_indices.shape[0] + 1)
        greater = (np.sum(surrogate_r > r) + 1) / (perm_indices.shape[0] + 1)
        p = 2 * min(less, greater)
    return r, surrogate_r, p


def fdr_correction(pvals: Dict[str, float]) -> Dict[str, float]:
    """
    Perform FDR correction on a dictionary of p-values.
    """
    p = np.array(list(pvals.values()))
    fdr_p = scipy.stats.false_discovery_control(p, method="bh")
    return {metric: fdr_p[i] for i, metric in enumerate(pvals.keys())}


def df_pval(x: pd.DataFrame, y: pd.DataFrame, fdr: bool = True, n_rep: int = 9999) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the p-values and correlations between x and y.
    """
    p = pd.DataFrame(index=x.columns, columns=y.columns, dtype=float)
    r = pd.DataFrame(index=x.columns, columns=y.columns, dtype=float)
    n_regions = x.shape[0]
    gen = make_glasser_nulls(n_rep=n_rep) if n_regions == 360 else make_dk_nulls(n_rep=n_rep)

    for i, x_col in enumerate(x.columns.to_list()):
        for j, y_col in enumerate(y.columns.to_list()):
            if n_regions == 360:
                r.loc[x_col, y_col], _, p.loc[x_col, y_col] = spin_test_glasser(gen, x.values[:, i], y.values[:, j])
            else:
                r.loc[x_col, y_col], _, p.loc[x_col, y_col] = spin_test_dk(gen, x.values[:, i], y.values[:, j])
    # Correct for multiple comparisons (row-wise)
    if fdr:
        for i, x_col in enumerate(p.index.to_list()):
            p.loc[x_col, :] = scipy.stats.false_discovery_control(p.loc[x_col, :], method="bh")

    return p, r


def compute_multilinear(x: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Compute the multilinear regression between x and y.
    """
    y_cols = y.columns.to_list()
    n_rep = 9999
    gen = make_glasser_nulls(n_rep=n_rep) if x.shape[0] == 360 else make_dk_nulls(n_rep=n_rep)
    r_df = pd.DataFrame(index=y_cols, columns=["r"], dtype=float)
    p_df = pd.DataFrame(index=y_cols, columns=["p"], dtype=float)
    X = np.hstack((x.values, np.ones((x.shape[0], 1))))
    w = np.linalg.lstsq(X, y.values, rcond=None)[0]
    pred = np.dot(X, w)

    for idx, y_col in enumerate(y_cols):
        if x.shape[0] == 360:
            r_df.loc[y_col, "r"], _, p_df.loc[y_col, "p"] = spin_test_glasser(gen, y.values[:, idx], pred[:, idx], alt="greater")
        else:
            r_df.loc[y_col, "r"], _, p_df.loc[y_col, "p"] = spin_test_dk(gen, y.values[:, idx], pred[:, idx], alt="greater")

    p_df.loc[:, "p"] = scipy.stats.false_discovery_control(p_df.values.ravel(), method="bh")
    return r_df, p_df, pred


def compute_cv(x: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the cross-validated correlation between x and y.
    """
    if x.shape[0] == 360:
        pts = np.loadtxt("diffusion_neuromaps/atlases/glasser_spherical_pts.txt")
    else:
        pts = np.loadtxt("diffusion_neuromaps/atlases/dk_spherical_pts.txt")

    y_cols = y.columns.to_list()
    n_rep = 9999
    gen = make_glasser_nulls(n_rep=n_rep) if x.shape[0] == 360 else make_dk_nulls(n_rep=n_rep)
    train_r_df = y.copy(deep=True)
    test_r_df = y.copy(deep=True)
    for col_idx, y_col in enumerate(y_cols):
        for row_idx, y_row in enumerate(y.index.to_list()):
            # find 25% nearest neighbors (90 for glasser, 17 for dk) to select test set
            # Set the other 75% as the training set
            dist = cdist(pts, pts[row_idx, :].reshape(1, -1))
            nn_idx = np.argsort(dist, axis=0)[:x.shape[0] // 4].flatten()
            X_test = np.hstack((x.values[nn_idx, :], np.ones((x.shape[0] // 4, 1))))
            X_train = np.hstack((x.values[~np.isin(np.arange(x.shape[0]), nn_idx), :], np.ones((x.shape[0] - x.shape[0] // 4, 1))))
            y_test = y.values[nn_idx, col_idx]
            y_train = y.values[~np.isin(np.arange(x.shape[0]), nn_idx), col_idx]
            w = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            pred_train = np.dot(X_train, w)
            pred_test = np.dot(X_test, w)

            train_r_df.loc[y_row, y_col] = np.corrcoef(y_train, pred_train)[0, 1]
            test_r_df.loc[y_row, y_col] = np.corrcoef(y_test, pred_test)[0, 1]

    return train_r_df, test_r_df


def compute_dominance(x: pd.DataFrame, y: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Compute the dominance analysis between x and y.
    """
    x_cols = x.columns.to_list()
    y_cols = y.columns.to_list()
    dominance_df = pd.DataFrame(index=y_cols, columns=x_cols)
    for i, y_col in enumerate(y_cols):
        data = pd.concat([x, y[y_col]], axis=1)
        dominance_regression=Dominance(data=data, target=y_col, objective=1, data_format=0, top_k=top_k)
        dominance_regression.incremental_rsquare()
        res = dominance_regression.dominance_stats()
        print(res)
        dominance_df.loc[y_col, :] = res["Percentage Relative Importance"]
    dominance_df = dominance_df.astype(float)

    return dominance_df
