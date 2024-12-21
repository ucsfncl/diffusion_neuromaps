import numpy as np
import pandas as pd
import os
from enigmatoolbox.datasets import load_sc, load_fc, fetch_ahba
from tools import make_glasser_nulls, spin_test_glasser, make_dk_nulls, spin_test_dk, fdr_correction, spin_test_adj_dk, spin_test_adj_glasser
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns


def exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(b * x) + c


def regress_out_exp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p0 = [1, -0.05, -0.1]  # initial parameter guesses
    pars, _ = curve_fit(exponential, x, y, p0=p0, maxfev=10000)
    expfit = exponential(x, pars[0], pars[1], pars[2])
    return y - expfit


def regress_out_dist_glasser(x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    triu_indices = np.triu_indices(360, k=1)
    glasser_df = pd.read_table("diffusion_neuromaps/atlases/Glasser360_Atlas.txt", sep=",", index_col=0, header=0, encoding="utf-8")
    dist = squareform(pdist(glasser_df.loc[:, ["x-cog", "y-cog", "z-cog"]].values, metric='euclidean'))[triu_indices]

    ret = np.zeros(x.shape)
    ret.fill(np.nan)
    if mask is not None:
        ret[mask] = regress_out_exp(dist[mask], x[mask])
    else:
        ret = regress_out_exp(dist, x)
    return ret


def regress_out_dist_dk(x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    triu_indices = np.triu_indices(68, k=1)
    dk_coords = np.loadtxt("diffusion_neuromaps/atlases/dk_coords.txt")
    dist = squareform(pdist(dk_coords, metric='euclidean'))[triu_indices]

    ret = np.zeros(x.shape)
    ret.fill(np.nan)
    if mask is not None:
        ret[mask] = regress_out_exp(dist[mask], x[mask])
    else:
        ret = regress_out_exp(dist, x)
    return ret

if __name__ == "__main__":

    ### 1. Load the structural covariance network for each metric (Glasser and DK parcellation)
    metrics = [
        "fa", "ad", "rd", "md", "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
        "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]

    dti_metrics = ["fa", "ad", "rd", "md"]
    dki_dti_metrics = ["dki-fa", "dki-ad", "dki-rd", "dki-md"]
    dki_metrics = ["mk", "ak", "rk", "kfa", "mkt"]
    noddi_metrics = ["icvf", "odi", "isovf"]
    mapmri_metrics = ["msd", "qiv", "rtop", "rtap", "rtpp"]
    
    dk_triu_indices = np.triu_indices(68, k=1)
    glasser_triu_indices = np.triu_indices(360, k=1)
    hcp_dk = {}
    hcp_glasser = {}
    scn_dk = {}
    scn_centrality_dk = {}
    scn_glasser = {}
    scn_centrality_glasser = {}
    for metric in metrics + ["all"]:
        df = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "scn", f"{metric}_glasser_scn.csv"), index_col=0).values.astype(float)
        np.fill_diagonal(df, 0.0)
        scn_glasser[metric] = df[glasser_triu_indices]
        scn_centrality_glasser[metric] = df.sum(axis=0)
        df = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "dk", "scn", f"{metric}_dk_scn.csv"), index_col=0).values.astype(float)
        np.fill_diagonal(df, 0.0)
        scn_dk[metric] = df[dk_triu_indices]
        scn_centrality_dk[metric] = df.sum(axis=0)
    
    for metric in metrics:
        df = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "ave", f"{metric}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
        hcp_glasser[metric] = df
        df = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "dk", "ave", f"{metric}_dk_ave.csv"), index_col=0).values.ravel().astype(float)
        hcp_dk[metric] = df

    ### 2. Load the structural and functional connectivity matrices
    fc_ctx, ctx_labels, _, _ = load_fc(parcellation="glasser_360")
    sc_ctx, ctx_labels, _, _ = load_sc(parcellation="glasser_360")
    sc_mask = (sc_ctx != 0)
    sc_ctx[sc_mask] = (sc_ctx[sc_mask] - np.min(sc_ctx[sc_mask])) / (np.max(sc_ctx[sc_mask]) - np.min(sc_ctx[sc_mask]))
    np.fill_diagonal(sc_ctx, 0)
    np.fill_diagonal(fc_ctx, 0)
    sc_centrality = np.sum(sc_ctx, axis=0)
    fc_centrality = np.sum(fc_ctx, axis=0)

    ### 3. Load the PET data and compute the PET netowk
    receptor_names =["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                    "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                    "MOR", "KOR", "NET", "NMDA", "VAChT"]

    pet_df = np.zeros((360, len(receptor_names)))
    for idx, name in enumerate(receptor_names):
        map_df = pd.read_csv(os.path.join("diffusion_neuromaps", "data", "pet", f"{name}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
        pet_df[:, idx] = map_df

    pet_ctx = np.corrcoef(pet_df)
    np.fill_diagonal(pet_ctx, 0.0)
    pet_centrality = np.sum(pet_ctx, axis=0)

    ### 4. Compute the gene expression network
    genes = fetch_ahba()
    genes.set_index("label", inplace=True)
    genes = genes.loc[[label for label in genes.index if label.startswith("R_") or label.startswith("L_")], :]
    genes.index = [f"rh.{label[2:]}" if label.startswith("R_") else f"lh.{label[2:]}" for label in genes.index]
    # column-wise gene mask for nan values
    gene_mask = ~np.isnan(np.sum(genes.values, axis=1))
    gene_ctx = np.corrcoef(genes.values[gene_mask, :])
    np.fill_diagonal(gene_ctx, 0.0)
    gene_centrality = np.zeros(gene_mask.shape[0])
    gene_centrality.fill(np.nan)
    gene_centrality[gene_mask] = np.sum(gene_ctx, axis=0)

    ### 5. Compute the distance matrices
    glasser_df = pd.read_table("diffusion_neuromaps/atlases/Glasser360_Atlas.txt", sep=",", index_col=0, header=0, encoding="utf-8")
    glasser_dist = squareform(pdist(glasser_df.loc[:, ["x-cog", "y-cog", "z-cog"]].values, metric='euclidean'))[glasser_triu_indices]

    dk_coords = np.loadtxt("diffusion_neuromaps/atlases/dk_coords.txt")
    dk_dist = squareform(pdist(dk_coords, metric='euclidean'))[dk_triu_indices]

    ### 6. Compute the centrality r and p values
    r_sc_centrality = {}
    p_sc_centrality = {}
    r_fc_centrality = {}
    p_fc_centrality = {}
    r_pet_centrality = {}
    p_pet_centrality = {}
    r_gene_centrality = {}
    p_gene_centrality = {}

    gen_dk = make_dk_nulls()
    gen_glasser = make_glasser_nulls()

    for metric in metrics:
        r_sc_centrality[metric], _, p_sc_centrality[metric] = spin_test_glasser(gen_glasser, sc_centrality, hcp_glasser[metric])
        r_fc_centrality[metric], _, p_fc_centrality[metric] = spin_test_glasser(gen_glasser, fc_centrality, hcp_glasser[metric])
        r_pet_centrality[metric], _, p_pet_centrality[metric] = spin_test_glasser(gen_glasser, pet_centrality, hcp_glasser[metric])
        r_gene_centrality[metric], _, p_gene_centrality[metric] = spin_test_dk(gen_dk, gene_centrality, hcp_dk[metric])

    p_sc_centrality = fdr_correction(p_sc_centrality)
    p_fc_centrality = fdr_correction(p_fc_centrality)
    p_pet_centrality = fdr_correction(p_pet_centrality)
    p_gene_centrality = fdr_correction(p_gene_centrality)

    print(p_sc_centrality)
    print(p_fc_centrality)
    print(p_pet_centrality)
    print(p_gene_centrality)

    ### 7. Compute the scn r and p values
    r_sc = {}
    p_sc = {}
    r_fc = {}
    p_fc = {}
    r_pet = {}
    p_pet = {}
    r_gene = {}
    p_gene = {}

    sc_ctx[~sc_mask] = np.nan
    gene_ctx_full = np.zeros((68, 68))
    gene_ctx_full.fill(np.nan)
    gene_ctx_full[np.ix_(gene_mask, gene_mask)] = gene_ctx
    gene_ctx = gene_ctx_full

    sc_ctx = sc_ctx[glasser_triu_indices]
    fc_ctx = fc_ctx[glasser_triu_indices]
    pet_ctx = pet_ctx[glasser_triu_indices]
    gene_ctx = gene_ctx[dk_triu_indices]

    gene_mask = ~np.isnan(gene_ctx)
    sc_mask = ~np.isnan(sc_ctx)

    for metric in metrics + ["all"]:
        r_sc[metric], _, p_sc[metric] = spin_test_adj_glasser(gen_glasser, regress_out_dist_glasser(sc_ctx, sc_mask), regress_out_dist_glasser(scn_glasser[metric], sc_mask))
        r_fc[metric], _, p_fc[metric] = spin_test_adj_glasser(gen_glasser, regress_out_dist_glasser(fc_ctx), regress_out_dist_glasser(scn_glasser[metric]))
        r_pet[metric], _, p_pet[metric] = spin_test_adj_glasser(gen_glasser, regress_out_dist_glasser(pet_ctx), regress_out_dist_glasser(scn_glasser[metric]))
        r_gene[metric], _, p_gene[metric] = spin_test_adj_dk(gen_dk, regress_out_dist_dk(gene_ctx, gene_mask), regress_out_dist_dk(scn_dk[metric], gene_mask))

    p_sc = fdr_correction(p_sc)
    p_fc = fdr_correction(p_fc)
    p_pet = fdr_correction(p_pet)
    p_gene = fdr_correction(p_gene)

    print(p_sc)
    print(p_fc)
    print(p_pet)
    print(p_gene)

    df_centrality = pd.DataFrame(columns=["Metric", "Correlation", "Type"])
    df_centrality = pd.concat([df_centrality, pd.DataFrame({"Metric": metrics, "Correlation": [r_sc_centrality[metric] for metric in metrics], "Type": "SC"})], ignore_index=True)
    df_centrality = pd.concat([df_centrality, pd.DataFrame({"Metric": metrics, "Correlation": [r_fc_centrality[metric] for metric in metrics], "Type": "FC"})], ignore_index=True)
    df_centrality = pd.concat([df_centrality, pd.DataFrame({"Metric": metrics, "Correlation": [r_pet_centrality[metric] for metric in metrics], "Type": "PET"})], ignore_index=True)
    df_centrality = pd.concat([df_centrality, pd.DataFrame({"Metric": metrics, "Correlation": [r_gene_centrality[metric] for metric in metrics], "Type": "Gene"})], ignore_index=True)

    df_conn = pd.DataFrame(columns=["Metric", "Correlation", "Type"])
    df_conn = pd.concat([df_conn, pd.DataFrame({"Metric": metrics + ["all"], "Correlation": [r_sc[metric] for metric in metrics] + [r_sc["all"]], "Type": "SC"})], ignore_index=True)
    df_conn = pd.concat([df_conn, pd.DataFrame({"Metric": metrics + ["all"], "Correlation": [r_fc[metric] for metric in metrics] + [r_fc["all"]], "Type": "FC"})], ignore_index=True)
    df_conn = pd.concat([df_conn, pd.DataFrame({"Metric": metrics + ["all"], "Correlation": [r_pet[metric] for metric in metrics] + [r_pet["all"]], "Type": "PET"})], ignore_index=True)
    df_conn = pd.concat([df_conn, pd.DataFrame({"Metric": metrics + ["all"], "Correlation": [r_gene[metric] for metric in metrics] + [r_gene["all"]], "Type": "Gene"})], ignore_index=True)

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"wspace": 0.0, "hspace": 0.1}, sharex=True)
    ax[0] = sns.barplot(ax=ax[0], data=df_conn, x="Metric", y="Correlation", hue="Type", palette="Set1", hue_order=["FC", "SC", "Gene", "PET"])
    ax[0].legend(loc='upper center', ncol=4, title="", fontsize=14)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].set_xlabel("")
    ax[0].set_ylabel("Correlation with Similarity")
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    ax[0].axhline(0.0, color="black", linewidth=1.0, alpha=1.0)
    ax[1].axhline(0.0, color="black", linewidth=1.0, alpha=1.0)
    for i in range(len(metrics)):
        ax[0].axvline(i + 0.5, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax[1].axvline(i + 0.5, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

    ax[1] = sns.barplot(ax=ax[1], data=df_centrality, x="Metric", y="Correlation", hue="Type", palette="Set1", hue_order=["FC", "SC", "Gene", "PET"], legend=False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].set_xlabel("")
    ax[1].set_ylabel("Correlation with Hubness", fontsize=14)
    ax[1].set_xticks(np.arange(24), labels=[txt.upper() for txt in metrics + ["all"]], rotation=270, fontsize=14)
    for i, txt in enumerate(metrics + ["all"]):
        if txt in dti_metrics:
            ax[1].get_xticklabels()[i].set_color("red")
        elif txt in dki_dti_metrics:
            ax[1].get_xticklabels()[i].set_color("purple")
        elif txt in dki_metrics:
            ax[1].get_xticklabels()[i].set_color("orange")
        elif txt in noddi_metrics:
            ax[1].get_xticklabels()[i].set_color("blue")
        elif txt in mapmri_metrics:
            ax[1].get_xticklabels()[i].set_color("green")
        else:
            ax[1].get_xticklabels()[i].set_color("black")

    fig.tight_layout()
    fig.savefig(os.path.join("diffusion_neuromaps", "plots", "figs", "fig_4.png"), bbox_inches="tight", dpi=400, pad_inches=0.05)
    plt.close(fig)