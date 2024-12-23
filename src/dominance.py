# Perform dominancel analysis over the structural metrics
# Takes a while to run so I have split it up via SGE array jobs
# Adjust as needed to your job submission system
# Comment out parts 2, 3, 4 separately and run them as separate array jobs
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from tools import compute_dominance
top_k = 25

# ### 1. Load the structural metrics
# metric_path = os.path.join("diffusion_neuromaps", "data")
# glasser_df = pd.read_csv(os.path.join("diffusion_neuromaps", "atlases", "glasser_parc.csv"), index_col=0)
# metrics = [
#     "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
#     "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"
#     ]

# struct_df = pd.DataFrame(index=glasser_df.index, columns=metrics, dtype=float)
# for metric in metrics:
#     hcp_df = pd.read_csv(os.path.join(metric_path, "glasser", f"{metric}_glasser_ave.csv"), index_col=0)
#     struct_df[metric] = hcp_df


# ###  2. Load the meg power metrics and performs dominance analysis (7 jobs)
# metric_path = os.path.join("diffusion_neuromaps", "data")
# job_idx = int(os.getenv("SGE_TASK_ID", "1")) - 1
# meg_names = ["megdelta", "megtheta", "megalpha", "megbeta", "meggamma1", "meggamma2", "megtimescale"]
# meg_names = [meg_names[job_idx]]
# meg_df = pd.DataFrame(index=glasser_df.index, columns=meg_names, dtype=float)
# for name in meg_names:
#     meg_df[name] = pd.read_csv(os.path.join(metric_path, "glasser", f"{name}_glasser_ave.csv"), index_col=0)

# meg_res = compute_dominance(struct_df, meg_df, top_k=len(metrics))
# meg_res.to_csv(os.path.join(metric_path, "dominance", f"meg_{meg_names[0]}_dominance.csv"))


# ###  3. Load the PET receptor metrics and performs dominance analysis (20 jobs)
# metric_path = os.path.join("diffusion_neuromaps", "data")
# job_idx = int(os.getenv("SGE_TASK_ID", "1")) - 1
# pet_names =["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
#             "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
#             "MOR", "KOR", "NET", "NMDA", "VAChT"]

# pet_names = [pet_names[job_idx]]
# pet_df = pd.DataFrame(index=glasser_df.index, columns=pet_names, dtype=float)
# for name in pet_names:
#     pet_df[name] = pd.read_csv(os.path.join(metric_path, "glasser", f"{name}_glasser_ave.csv"), index_col=0)

# pet_res = compute_dominance(struct_df, pet_df, top_k=len(metrics))
# pet_res.to_csv(os.path.join(metric_path, "dominance", f"pet_{pet_names[0]}_dominance.csv"))


###  4. Load the DKT structural maps and the disorder maps and perform dominance analysis (11 jobs)
metric_path = os.path.join("diffusion_neuromaps", "data")
job_idx = int(os.getenv("SGE_TASK_ID", "1")) - 1
with open("diffusion_neuromaps/atlases/fslr_dk.json", "r") as f:
    fslr_labels = json.load(f)
fslr_labels = {int(k): v for k, v in fslr_labels.items()}
dk_regions = [f"lh.{val}" for val in fslr_labels.values()] + [f"rh.{val}" for val in fslr_labels.values()]

metrics = [
    "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
    "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "myl"]

struct_df = pd.DataFrame(index=dk_regions, columns=metrics, dtype=float)
for metric in metrics:
    hcp_df = pd.read_csv(os.path.join(metric_path, "dk", "ave", f"{metric}_dk_ave.csv"), index_col=0)
    struct_df[metric] = hcp_df

disorder_names = ["adhd_ct", "adhd_sa", "bd_ct", "bd_sa", "mdd_ct", "mdd_sa", "ocd_ct", "ocd_sa", "scz_ct", "scz_sa", "asd_ct"]
disorder_names = [disorder_names[job_idx]]
disorder_df = pd.DataFrame(index=dk_regions, columns=disorder_names, dtype=float)
for name in disorder_names:
    disorder_df[name] = pd.read_csv(os.path.join(metric_path, "disorders", f"{name}_dk_ave.csv"), index_col=0)

disorder_res = compute_dominance(struct_df, disorder_df, top_k=len(metrics))
disorder_res.to_csv(os.path.join(metric_path, "dominance", f"disorder_{disorder_names[0]}_dominance.csv"))