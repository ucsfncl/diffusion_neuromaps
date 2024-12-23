import numpy as np
import pandas as pd
import os
from enigmatoolbox.datasets import load_summary_stats
import json


def load_disorder_data(disorder: str, map_str: str) -> pd.DataFrame:
    with open("diffusion_neuromaps/atlases/fslr_dk.json", "r") as f:
        fslr_labels = json.load(f)
    fslr_labels = {int(k): v for k, v in fslr_labels.items()}
    dkt_regions = [f"lh.{val}" for val in fslr_labels.values()] + [f"rh.{val}" for val in fslr_labels.values()]

    sum_stats = load_summary_stats(disorder)
    CT = sum_stats[map_str]
    d = {CT.loc[i, "Structure"]: CT.loc[i, "d_icv"] for i in range(CT.shape[0])}
    d_disorder = {key.replace("L_", "lh.").replace("R_", "rh."): value for key, value in d.items()}
    df_disorder = pd.DataFrame(index=dkt_regions, columns=["d"])
    for region in dkt_regions:
        df_disorder.loc[region, "d"] = d_disorder[region]
    return df_disorder


###  1. Load Psychiatric Disorder vs Control Data
metric_path = os.path.join("diffusion_neuromaps", "data")

adhd_ct = load_disorder_data('adhd', 'CortThick_case_vs_controls_adult')
adhd_sa = load_disorder_data('adhd', 'CortSurf_case_vs_controls_adult')
asd_ct = load_disorder_data('asd', 'CortThick_case_vs_controls_meta_analysis')
bd_ct = load_disorder_data('bipolar', 'CortThick_case_vs_controls_adult')
bd_sa = load_disorder_data('bipolar', 'CortSurf_case_vs_controls_adult')
mdd_ct = load_disorder_data('depression', 'CortThick_case_vs_controls_adult')
mdd_sa = load_disorder_data('depression', 'CortSurf_case_vs_controls_adult')
ocd_ct = load_disorder_data('ocd', 'CortThick_case_vs_controls_adult')
ocd_sa = load_disorder_data('ocd', 'CortSurf_case_vs_controls_adult')
scz_ct = load_disorder_data('schizophrenia', 'CortThick_case_vs_controls')
scz_sa = load_disorder_data('schizophrenia', 'CortSurf_case_vs_controls')

adhd_ct.to_csv(os.path.join(metric_path, "disorders", "adhd_ct_dk_ave.csv"))
adhd_sa.to_csv(os.path.join(metric_path, "disorders", "adhd_sa_dk_ave.csv"))
asd_ct.to_csv(os.path.join(metric_path, "disorders", "asd_ct_dk_ave.csv"))
bd_ct.to_csv(os.path.join(metric_path, "disorders", "bd_ct_dk_ave.csv"))
bd_sa.to_csv(os.path.join(metric_path, "disorders", "bd_sa_dk_ave.csv"))
mdd_ct.to_csv(os.path.join(metric_path, "disorders", "mdd_ct_dk_ave.csv"))
mdd_sa.to_csv(os.path.join(metric_path, "disorders", "mdd_sa_dk_ave.csv"))
ocd_ct.to_csv(os.path.join(metric_path, "disorders", "ocd_ct_dk_ave.csv"))
ocd_sa.to_csv(os.path.join(metric_path, "disorders", "ocd_sa_dk_ave.csv"))
scz_ct.to_csv(os.path.join(metric_path, "disorders", "scz_ct_dk_ave.csv"))
scz_sa.to_csv(os.path.join(metric_path, "disorders", "scz_sa_dk_ave.csv"))
