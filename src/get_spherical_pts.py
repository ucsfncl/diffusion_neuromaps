from nilearn.surface import load_surf_data
from brainspace.mesh.mesh_elements import get_points 
from brainspace.datasets import load_conte69
import json
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd


lh_sphere, rh_sphere = load_conte69(as_sphere=True)
lh_pts = get_points(lh_sphere)
rh_pts = get_points(rh_sphere)

lh_dk = load_surf_data("diffusion_neuromaps/atlases/Desikan.32k.L.label.gii")
rh_dk = load_surf_data("diffusion_neuromaps/atlases/Desikan.32k.R.label.gii")
with open("microstructure/atlases/fslr_dkt.json", "r") as f:
    fslr_labels = json.load(f)
fslr_labels = {int(k): v for k, v in fslr_labels.items()}

ret_lh = {}
ret_rh = {}
for key, val in fslr_labels.items():
    lh_dist = squareform(pdist(lh_pts[lh_dk == key])) ** 2
    rh_dist = squareform(pdist(rh_pts[rh_dk == key])) ** 2

    lh_min = np.argmin(np.sum(lh_dist, axis=0))
    rh_min = np.argmin(np.sum(rh_dist, axis=0))

    ret_lh[f"lh.{val}"] = lh_pts[lh_dk == key][lh_min]
    ret_rh[f"rh.{val}"] = rh_pts[rh_dk == key][rh_min]

ret_lh = np.array([val for val in ret_lh.values()])
ret_rh = np.array([val for val in ret_rh.values()])
ret = np.vstack([ret_lh, ret_rh])
np.savetxt("diffusion_neuromaps/atlases/dk_spherical_pts.txt", ret)

lh_sphere, rh_sphere = load_conte69(as_sphere=True)
lh_pts = get_points(lh_sphere)
rh_pts = get_points(rh_sphere)

lh_glasser = load_surf_data("diffusion_neuromaps/atlases/Glasser_2016.32k.L.label.gii")
rh_glasser = load_surf_data("diffusion_neuromaps/atlases/Glasser_2016.32k.R.label.gii")
coords = np.zeros((360, 3))

for i in range(1, 181):
    # compute pairwise distances between every point in the region
    lh_dist = squareform(pdist(lh_pts[lh_glasser == i])) ** 2
    rh_dist = squareform(pdist(rh_pts[rh_glasser == i])) ** 2
    # find the point with the minimum sum of squared distances
    lh_min = np.argmin(np.sum(lh_dist ** 2, axis=0))
    rh_min = np.argmin(np.sum(rh_dist ** 2, axis=0))
    coords[i - 1, :] = lh_pts[lh_glasser == i][lh_min]
    coords[i + 179, :] = rh_pts[rh_glasser == i][rh_min]

np.savetxt("diffusion_neuromaps/atlases/glasser_spherical_pts.txt", coords)


lh_sphere, rh_sphere = load_conte69(as_sphere=True)
lh_pts = get_points(lh_sphere)
rh_pts = get_points(rh_sphere)

lh_dkt = load_surf_data("microstructure/atlases/Desikan.32k.L.label.gii")
rh_dkt = load_surf_data("microstructure/atlases/Desikan.32k.R.label.gii")
with open("microstructure/atlases/fslr_dkt.json", "r") as f:
    fslr_labels = json.load(f)
fslr_labels = {int(k): v for k, v in fslr_labels.items()}

ret_lh = {}
ret_rh = {}
for key, val in fslr_labels.items():
    ret_lh[f"lh.{val}"] = np.mean(lh_pts[np.where(lh_dkt == key)[0], :], axis=0)
    ret_rh[f"rh.{val}"] = np.mean(rh_pts[np.where(rh_dkt == key)[0], :], axis=0)

ret_lh = np.array([val for val in ret_lh.values()])
ret_rh = np.array([val for val in ret_rh.values()])
ret = np.vstack([ret_lh, ret_rh])
np.savetxt("diffusion_neuromaps/atlases/dk_coords.txt", ret)