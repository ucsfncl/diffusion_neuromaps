import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    metrics = [
        "fa", "ad", "rd", "md", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
        "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]
    
    ### 1. Compute the structural covariance network for each metric (Glasser parcellation)

    hcp_struct = []
    for metric in metrics:
        metric_df = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "subject", f"{metric}_glasser.csv"), index_col=0)
        metric_vals = metric_df.values
        metric_vals = (metric_vals - metric_vals.mean(axis=0)) / metric_vals.std(axis=0)
        hcp_struct.append(metric_vals)
        scn_df = np.corrcoef(metric_vals.T)
        np.fill_diagonal(scn_df, 1.0)
        scn_df = pd.DataFrame(scn_df, index=metric_df.columns, columns=metric_df.columns)
        scn_df.to_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "scn", f"{metric}_glasser_scn.csv"))

    hcp_struct = np.concatenate(hcp_struct, axis=0)
    hcp_struct_scn = np.corrcoef(hcp_struct.T)
    np.fill_diagonal(hcp_struct_scn, 1.0)
    hcp_struct_scn = pd.DataFrame(hcp_struct_scn, index=metric_df.columns, columns=metric_df.columns)
    hcp_struct_scn.to_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "scn", "all_glasser_scn.csv"))

    ### 2. Compute the structural covariance network for each metric (DK parcellation)

    hcp_struct = []
    for metric in metrics:
        metric_df = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "dk", "subject", f"{metric}_dk.csv"), index_col=0)
        metric_vals = metric_df.values
        metric_vals = (metric_vals - metric_vals.mean(axis=0)) / metric_vals.std(axis=0)
        hcp_struct.append(metric_vals)
        scn_df = np.corrcoef(metric_vals.T)
        np.fill_diagonal(scn_df, 1.0)
        scn_df = pd.DataFrame(scn_df, index=metric_df.columns, columns=metric_df.columns)
        scn_df.to_csv(os.path.join("diffusion_neuromaps", "data", "dk", "scn", f"{metric}_dk_scn.csv"))
    
    hcp_struct = np.concatenate(hcp_struct, axis=0)
    hcp_struct_scn = np.corrcoef(hcp_struct.T)
    np.fill_diagonal(hcp_struct_scn, 1.0)
    hcp_struct_scn = pd.DataFrame(hcp_struct_scn, index=metric_df.columns, columns=metric_df.columns)
    hcp_struct_scn.to_csv(os.path.join("diffusion_neuromaps", "data", "dk", "scn", "all_dk_scn.csv"))
    
