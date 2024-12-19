from typing import Tuple
import os

import numpy as np
import pandas as pd

def compute_li(df: pd.DataFrame) -> pd.DataFrame:
    left_idx = [idx for idx in df.index if idx.endswith("_L")]
    right_idx = [idx for idx in df.index if idx.endswith("_R")]
    li_idx = [idx[:-2] for idx in left_idx]
    df_lh = df.loc[left_idx].values.ravel().astype(float)
    df_rh = df.loc[right_idx].values.ravel().astype(float)
    li_data = (df_lh - df_rh) / (df_lh + df_rh)
    li_df = pd.DataFrame(li_data, index=li_idx, columns=[df.columns[0] + "_li"])
    li_df.index.name = "region"
    return li_df


def compute_retest_cov(test_df: pd.DataFrame, retest_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the coefficient of variation and standard error from the retest data.
    """
    hcp_mean = (test_df + retest_df) / 2
    hcp_std = (((test_df - hcp_mean) ** 2 + (retest_df - hcp_mean) ** 2) / 2) ** 0.5
    cov = hcp_std.mean(axis=0) / hcp_mean.mean(axis=0) * 100
    hcp_std_mean = hcp_std.mean(axis=0)
    cov = cov.to_frame()
    cov.index.name = "region"
    hcp_std_mean = hcp_std_mean.to_frame()
    hcp_std_mean.index.name = "region"
    return cov, hcp_std_mean

def compute_intersubject_cov_and_icc(test_df: pd.DataFrame, std_err_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the intersubject coefficient of variation and intraclass correlation coefficient.
    """
    hcp_var = test_df.var(axis=0)
    hcp_icc = (hcp_var - std_err_df.values.ravel().astype(float) ** 2) / hcp_var
    hcp_icc = hcp_icc.clip(0.0, 1.0)
    intersubject_cov = (hcp_var - std_err_df.values.ravel().astype(float) ** 2).clip(0, None) ** 0.5 / test_df.mean(axis=0).values.ravel().astype(float) * 100
    hcp_icc = hcp_icc.to_frame()
    hcp_icc.index.name = "region"
    intersubject_cov = intersubject_cov.to_frame()
    intersubject_cov.index.name = "region"
    return intersubject_cov, hcp_icc

if __name__ == "__main__":
    retest_subjects = np.loadtxt("diffusion_neuromaps/data/hcp_retest_qc_subjects.txt", dtype=int).tolist()
    subjects = np.loadtxt("diffusion_neuromaps/data/hcp_qc_subjects.txt", dtype=int).tolist()
    metrics = [
        "fa", "ad", "rd", "md", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
        "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]

    for metric in metrics:
        hcp_test = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "subject", f"{metric}_glasser.csv"), index_col=0)
        hcp_retest = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "retest_subject", f"{metric}_glasser.csv"), index_col=0)
        hcp_ave = hcp_test.mean(axis=0).to_frame()
        hcp_ave.index.name = "region"
        hcp_ave.rename(columns={0: metric}, inplace=True)

        hcp_retest_ave = hcp_retest.mean(axis=0).to_frame()
        hcp_retest_ave.index.name = "region"
        hcp_retest_ave.rename(columns={0: metric}, inplace=True)

        hcp_li = compute_li(hcp_ave)

        hcp_retest_cov, hcp_std_err = compute_retest_cov(hcp_test.loc[retest_subjects, :], hcp_retest.loc[retest_subjects, :])
        hcp_retest_cov.rename(columns={0: f"{metric}_retest_cov"}, inplace=True)
        hcp_std_err.rename(columns={0: f"{metric}_std_err"}, inplace=True)

        intersubject_cov, hcp_icc = compute_intersubject_cov_and_icc(hcp_test, hcp_std_err)
        intersubject_cov.rename(columns={0: f"{metric}_intersubject_cov"}, inplace=True)
        hcp_icc.rename(columns={0: f"{metric}_icc"}, inplace=True)

        hcp_ave.to_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "ave", f"{metric}_glasser_ave.csv"))
        hcp_retest_ave.to_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "retest_ave", f"{metric}_glasser_retest_ave.csv"))
        hcp_li.to_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "li", f"{metric}_glasser_li.csv"))
        hcp_retest_cov.to_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "retest_cov", f"{metric}_glasser_retest_cov.csv"))
        intersubject_cov.to_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "intersubject_cov", f"{metric}_glasser_intersubject_cov.csv"))
        hcp_icc.to_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "icc", f"{metric}_glasser_icc.csv"))
