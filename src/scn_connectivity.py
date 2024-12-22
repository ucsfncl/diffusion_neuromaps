import numpy as np
import pandas as pd
import os
from enigmatoolbox.datasets import load_sc
from tools import make_glasser_nulls, fdr_correction
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import scipy
import scipy.stats


if __name__ == "__main__":

    metrics = [
        "fa", "ad", "rd", "md", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
        "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]
    
    glasser_triu_indices = np.triu_indices(360, k=1)

    scn_glasser = {}
    for metric in metrics + ["all"]:
        df = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "scn", f"{metric}_glasser_scn.csv"), index_col=0).values.astype(float)
        np.fill_diagonal(df, 0.0)
        scn_glasser[metric] = df[glasser_triu_indices]

    sc_ctx, ctx_labels, _, _ = load_sc(parcellation="glasser_360")
    sc_ctx[sc_ctx == 0] = np.nan
    sc_ctx = sc_ctx[glasser_triu_indices]

    conn_mask = ~np.isnan(sc_ctx)
    not_conn_mask = np.isnan(sc_ctx)

    glasser_yeo = pd.read_csv("diffusion_neuromaps/atlases/glasser_parc.csv", index_col=0)["yeo"].values.ravel()
    yeo_labels = np.unique(glasser_yeo)
    within_mask = np.zeros((360, 360), dtype=bool)
    between_mask = np.zeros((360, 360), dtype=bool)
    for label in yeo_labels:
        within_mask[np.ix_(glasser_yeo == label, glasser_yeo == label)] = True
        between_mask[np.ix_(glasser_yeo == label, glasser_yeo != label)] = True

    within_mask = within_mask[glasser_triu_indices]
    between_mask = between_mask[glasser_triu_indices]
    gen = make_glasser_nulls()
    p_sc = {}
    t_sc = {}
    p_fc = {}
    t_fc = {}
    for metric in metrics + ["all"]:
        t_stat = scipy.stats.ttest_ind(scn_glasser[metric][conn_mask], scn_glasser[metric][not_conn_mask], equal_var=False).statistic
        x_mat = np.zeros((360, 360))
        triu_indices = np.triu_indices(360, k=1)
        tril_indices = np.tril_indices(360, k=-1)
        x_mat[triu_indices] = scn_glasser[metric]
        x_mat[tril_indices] = scn_glasser[metric]
        perm_indices = np.hstack(gen.randomize(np.arange(180), np.arange(180, 360)))
        surrogate_t = np.zeros(perm_indices.shape[0])   
        for i in range(perm_indices.shape[0]):
            surrogate_t[i] = scipy.stats.ttest_ind(x_mat[perm_indices[i], :][:, perm_indices[i]][glasser_triu_indices][conn_mask], x_mat[perm_indices[i], :][:, perm_indices[i]][glasser_triu_indices][not_conn_mask], equal_var=False).statistic
        p = (np.sum(surrogate_t > t_stat) + 1) / (perm_indices.shape[0] + 1)
        p_sc[metric] = p
        t_sc[metric] = t_stat

    p_sc = fdr_correction(p_sc)

    for metric in metrics + ["all"]:
        t_stat = scipy.stats.ttest_ind(scn_glasser[metric][within_mask], scn_glasser[metric][between_mask], equal_var=False).statistic
        x_mat = np.zeros((360, 360))
        triu_indices = np.triu_indices(360, k=1)
        tril_indices = np.tril_indices(360, k=-1)
        x_mat[triu_indices] = scn_glasser[metric]
        x_mat[tril_indices] = scn_glasser[metric]
        perm_indices = np.hstack(gen.randomize(np.arange(180), np.arange(180, 360)))
        surrogate_t = np.zeros(perm_indices.shape[0])
        for i in range(perm_indices.shape[0]):
            surrogate_t[i] = scipy.stats.ttest_ind(x_mat[perm_indices[i], :][:, perm_indices[i]][glasser_triu_indices][within_mask], x_mat[perm_indices[i], :][:, perm_indices[i]][glasser_triu_indices][between_mask], equal_var=False).statistic
        p = (np.sum(surrogate_t > t_stat) + 1) / (perm_indices.shape[0] + 1)
        p_fc[metric] = p
        t_fc[metric] = t_stat

    p_fc = fdr_correction(p_fc)

    print(p_sc)
    print(p_fc)
    print(t_sc)
    print(t_fc)

    sc_label = np.where(conn_mask, "Yes", "No")
    df_sc = pd.DataFrame(columns=["Similarity", "Metric", "Connected"])
    for metric in metrics + ["all"]:
        if df_sc.shape[0] == 0:
            df_sc = pd.DataFrame({"Similarity": scn_glasser[metric], "Metric": metric.upper(), "Connected": sc_label})
        else:
            df_sc = pd.concat((df_sc, pd.DataFrame({"Similarity": scn_glasser[metric], "Metric": metric.upper(), "Connected": sc_label})))

    fc_label = np.where(within_mask, "Yes", "No")
    df_fc = pd.DataFrame(columns=["Similarity", "Metric", "Connected"])
    for metric in metrics + ["all"]:
        if df_fc.shape[0] == 0:
            df_fc = pd.DataFrame({"Similarity": scn_glasser[metric], "Metric": metric.upper(), "Connected": fc_label})
        else:
            df_fc = pd.concat((df_fc, pd.DataFrame({"Similarity": scn_glasser[metric], "Metric": metric.upper(), "Connected": fc_label})))
        

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"hspace": 0.0, "wspace": 0.0}, sharex=True, sharey=True)
    ax[0] = sns.boxplot(data=df_sc, x="Metric", y="Similarity", hue="Connected", ax=ax[0], fill=False, showfliers=False, hue_order=["Yes", "No"], palette=["red", "blue"])
    ax[0].legend(loc="upper left", title="Connected", ncols=2)
    ax[0].set_xlabel("")
    ax[0].set_ylabel("Structural Connectivity")
    ax[0].set_ylim([0.0, 1.0])
    ax[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ax[1] = sns.boxplot(data=df_fc, x="Metric", y="Similarity", hue="Connected", ax=ax[1], fill=False, showfliers=False, hue_order=["Yes", "No"], palette=["red", "blue"])
    ax[1].legend(loc="upper left", title="Within Network", ncols=2)
    ax[1].set_xlabel("")
    ax[1].set_ylabel("Functional Connectivity")
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].set_xticks(np.arange(len(metrics) + 1), labels=[txt.upper() for txt in metrics + ["all"]], rotation=270)
    ax[1].set_ylim([0.0, 1.0])
    ax[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    for i in range(4):
        ax[1].get_xticklabels()[i].set_color("red")
    for i in range(4, 8):
        ax[1].get_xticklabels()[i].set_color("purple")
    for i in range(8, 13):
        ax[1].get_xticklabels()[i].set_color("orange")
    for i in range(13, 16):
        ax[1].get_xticklabels()[i].set_color("blue")
    for i in range(16, 21):
        ax[1].get_xticklabels()[i].set_color("green")

    fig.tight_layout()
    fig.savefig("diffusion_neuromaps/plots/figs/scn_connectivity.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)









    