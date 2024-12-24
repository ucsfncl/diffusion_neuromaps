import numpy as np
import pandas as pd
import os

from tools import df_pval, make_glasser_nulls, spin_test_glasser, fdr_correction

### 1. Load the scores
glasser_df = pd.read_csv(os.path.join("diffusion_neuromaps", "atlases", "glasser_parc.csv"), index_col=0)
metric_path = os.path.join("diffusion_neuromaps", "data")
pet_pls_struct_score = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "pls", "pet_pls_struct_score_glasser_ave.csv"), index_col=0)
pet_pls_receptor_score = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "pls", "pet_pls_receptor_score_glasser_ave.csv"), index_col=0)
neuro_pls_struct_score = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "pls", "neuro_pls_struct_score_glasser_ave.csv"), index_col=0)
neuro_pls_cog_score = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "pls", "neuro_pls_cog_score_glasser_ave.csv"), index_col=0)


scores_df = pd.concat([pet_pls_struct_score, pet_pls_receptor_score, neuro_pls_struct_score, neuro_pls_cog_score], axis=1)
score_cols = ["pet_struct", "pet_receptor", "neuro_struct", "neuro_cog"]
scores_df.columns = score_cols

### 2. Load the other neuromaps and perform univariate analysis
other_names = ["SAaxis", "FChomology", "evoexp", "fcgradient01", "genepc1", "arealscaling",
               "cbf", "cbv", "cmr02", "cmrglc", "intersubjvar", "SV2A", 
               "bielschowsky", "blockface", "pd", "r1", "r2-star", "parvalbumin", "thionin"]

other_df = pd.DataFrame(index=glasser_df.index, columns=other_names, dtype=float)
for name in other_names:
    other_val = pd.read_csv(os.path.join(metric_path, "neuromaps", f"{name}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
    other_df[name] = other_val

p_other, r_other = df_pval(other_df, scores_df)
for name in other_names:
    for score_col in score_cols:
        if p_other.loc[name, score_col] < 0.05:
            print(f"{name} vs {score_col}: {r_other.loc[name, score_col]:.3f}, {p_other.loc[name, score_col]:.4f}")

### 2. Look at the correlations between the scores
gen = make_glasser_nulls(n_rep=9999)
p = {}
r = {}
for i in range(4):
    for j in range(i):
        col_i = scores_df.columns[i]
        col_j = scores_df.columns[j]
        val_i = scores_df.values[:, i]
        val_j = scores_df.values[:, j]
        r[f"{col_i} vs {col_j}"], null_r, p[f"{col_i} vs {col_j}"] = spin_test_glasser(gen, val_i, val_j)

p = fdr_correction(p)

for key in p.keys():
    if p[key] < 0.05:
        print(f"{key}: {r[key]:.3f}, {p[key]:.4f}")
