import os

import numpy as np
import pandas as pd


if __name__ == "__main__":
    subjects = np.loadtxt("diffusion_neuromaps/data/hcp_qc_subjects.txt", dtype=int).tolist()
    metrics = [
        "fa", "ad", "rd", "md", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
        "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]

    os.makedirs("diffusion_neuromaps/data/dk/ave", exist_ok=True)

    for metric in metrics:
        hcp_test = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "dk", "subject", f"{metric}_dk.csv"), index_col=0)
        hcp_ave = hcp_test.mean(axis=0).to_frame()
        hcp_ave.index.name = "region"
        hcp_ave.rename(columns={0: metric}, inplace=True)
        hcp_ave.to_csv(os.path.join("diffusion_neuromaps", "data", "dk", "ave", f"{metric}_dk_ave.csv"))