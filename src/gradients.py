import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tools import plot_glasser_map, df_pval


def matrix_laplacian(x):
    d = np.sum(x, axis=0)
    d_mat = np.diag(d)
    L = d_mat - x
    # return L
    d_sqrt_inv = 1 / np.sqrt(d)
    L_norm = np.einsum('ij,i,j->ij', L, d_sqrt_inv, d_sqrt_inv)
    return L_norm


### 1. Load the structural metrics
metric_path = os.path.join("diffusion_neuromaps", "data")
glasser_df = pd.read_csv(os.path.join("diffusion_neuromaps", "atlases", "glasser_parc.csv"), index_col=0)
metrics = [
    "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
    "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"]
    
struct_df = pd.DataFrame(index=glasser_df.index, columns=metrics, dtype=float)
for metric in metrics:
    hcp_val = pd.read_csv(os.path.join(metric_path, "glasser", "ave", f"{metric}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
    hcp_val = (hcp_val - hcp_val.mean()) / hcp_val.std()
    struct_df[metric] = hcp_val

X = np.zeros((len(metrics), 962, 360))
for i, metric in enumerate(metrics):
    hcp_df = pd.read_csv(os.path.join(metric_path, "glasser", "subject", f"{metric}_glasser.csv"), index_col=0).astype(float).values
    hcp_df = (hcp_df - np.mean(hcp_df, axis=0)) / np.std(hcp_df, axis=0)
    X[i, :, :] = hcp_df
X = X.reshape(-1, X.shape[-1]).T
R = np.corrcoef(X)
np.fill_diagonal(R, 0.0)
R = np.clip(R, 0, 1)
L = matrix_laplacian(R)
evals, evecs = np.linalg.eigh(L)
evecs *= -1
evecs = (evecs - evecs.mean(axis=0)) / evecs.std(axis=0)

for i in range(1, 11):
    plot_glasser_map(evecs[:, i], size=(1600, 400), cmap="RdBu_r", label_text=None, color_bar=False,
                     nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
                     screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", f"evec_{i}.png"))
    
    df_grad = pd.DataFrame(evecs[:, i], index=glasser_df.index, columns=[f"grad{i}"])
    df_grad.to_csv(os.path.join("diffusion_neuromaps", "data", "glasser", "ave", f"grad{i}_glasser_ave.csv"))


### 2. Load the other neuromaps and perform univariate analysis with metrics
other_names = ["SAaxis", "FChomology", "evoexp", "fcgradient01", "genepc1", "arealscaling",
               "cbf", "cbv", "cmr02", "cmrglc", "intersubjvar", "SV2A", 
               "bielschowsky", "blockface", "pd", "r1", "r2-star", "parvalbumin", "thionin"]

other_df = pd.DataFrame(index=glasser_df.index, columns=other_names, dtype=float)
for name in other_names:
    other_val = pd.read_csv(os.path.join(metric_path, "neuromaps", f"{name}_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
    other_df[name] = other_val
    plot_glasser_map(other_val, size=(1600, 400), cmap="Reds", label_text=None, color_bar=False,
                     nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
                     screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", f"{name}_glasser_ave.png"))


p_other, r_other = df_pval(other_df, struct_df)
for name in other_names:
    for metric in metrics:
        if p_other.loc[name, metric] < 0.05:
            print(f"{name} vs {metric}: {r_other.loc[name, metric]:.3f}, {p_other.loc[name, metric]:.4f}")

### 2. Perform univariate analysis with gradients
grad_df = pd.DataFrame(index=glasser_df.index, columns=[f"grad1", "grad2"], dtype=float)
grad_df["grad1"] = evecs[:, 1]
grad_df["grad2"] = evecs[:, 2]

p_grad, r_grad = df_pval(other_df, grad_df)
for name in other_names:
    for grad in ["grad1", "grad2"]:
        if p_grad.loc[name, grad] < 0.05:
            print(f"{name} vs {grad}: {r_grad.loc[name, grad]:.3f}, {p_grad.loc[name, grad]:.4f}")

### 3. Plot the correlation between grad2 and SA axis
sa_axis = pd.read_csv(os.path.join(metric_path, "neuromaps", "SAaxis_glasser_ave.csv"), index_col=0).values.ravel().astype(float)
sa_axis = (sa_axis - sa_axis.mean()) / sa_axis.std()

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.scatter(sa_axis, evecs[:, 2], color="blue", s=10, alpha=0.5)
ax.annotate("r=0.678\np=0.004", xy=(0.25, 0.85), xycoords="axes fraction", va="center", ha="center", fontsize=14)
m, b = np.polyfit(sa_axis, evecs[:, 2], 1)
ax.plot(sa_axis, m * sa_axis + b, color="red", linewidth=2)
ax.set_xlabel("SA Axis", fontsize=14)
ax.set_ylabel('SG2', fontsize=14)
ax.set_yticks([-2.0, 0.0, 2.0])
ax.set_xticks([-2.0,  0.0, 2.0])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig("diffusion_neuromaps/plots/figs/sa_axis_grad2_corr.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
plt.close()

### 4. Plot stratification of SG1 and SG2 along SA axis, mesulam, and economo
economo = glasser_df["economo"].values.ravel()
mesulam = glasser_df["mesulam"].values.ravel()
sa = glasser_df["sa"].values.ravel()

df_economo = pd.DataFrame(columns=["Metric", "Value", "Region"])
df_mesulam = pd.DataFrame(columns=["Metric", "Value", "Region"])
df_sa_axis = pd.DataFrame(columns=["Metric", "Value", "Region"])

df_economo = pd.concat([df_economo, pd.DataFrame({"Metric": ["SG1"] * len(economo), "Value": evecs[:, 1], "Region": economo})], axis=0)
df_economo = pd.concat([df_economo, pd.DataFrame({"Metric": ["SG2"] * len(economo), "Value": evecs[:, 2], "Region": economo})], axis=0)

df_mesulam = pd.concat([df_mesulam, pd.DataFrame({"Metric": ["SG1"] * len(mesulam), "Value": evecs[:, 1], "Region": mesulam})], axis=0)
df_mesulam = pd.concat([df_mesulam, pd.DataFrame({"Metric": ["SG2"] * len(mesulam), "Value": evecs[:, 2], "Region": mesulam})], axis=0)

df_sa_axis = pd.concat([df_sa_axis, pd.DataFrame({"Metric": ["SG1"] * len(sa), "Value": evecs[:, 1], "Region": sa})], axis=0)
df_sa_axis = pd.concat([df_sa_axis, pd.DataFrame({"Metric": ["SG2"] * len(sa), "Value": evecs[:, 2], "Region": sa})], axis=0)

df_sa_axis.drop(df_sa_axis[df_sa_axis["Region"] == "middle"].index, inplace=True)

economo_pal = sns.color_palette("tab10", 5)
mesulam_pal = sns.color_palette("tab10")[5:9]
sa_pal = {"association": "red", "sensorimotor": "blue"}

fig, ax = plt.subplots(1, 1, figsize=(3, 4), gridspec_kw={"hspace": 0.0, "wspace": 0.0})
ax = sns.boxplot(ax=ax, data=df_economo, x="Metric", y="Value", hue="Region", order=["SG1", "SG2"], hue_order=["agranular", "frontal", "parietal", "polar", "granular"], palette=economo_pal, showfliers=False)
ax.legend(loc='upper right', ncol=1, title="", bbox_to_anchor=(1.0, 1.15), fontsize=14)
ax.set_xticks(np.arange(2), labels=["SG1", "SG2"], fontsize=14)
ax.set_yticks([-2.0, 0.0, 2.0])
ax.set_ylim(None, 5.25)
ax.set_ylabel("Von Economo", fontsize=14)
ax.set_xlabel("")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join("diffusion_neuromaps", "plots", "figs", "gradient_economo.png"), bbox_inches="tight", dpi=400, pad_inches=0.05)
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(3, 4), gridspec_kw={"hspace": 0.0, "wspace": 0.0})
ax = sns.boxplot(ax=ax, data=df_mesulam, x="Metric", y="Value", hue="Region", order=["SG1", "SG2"], hue_order=["paralimbic", "heteromodal", "unimodal", "idiotypic"], palette=mesulam_pal, showfliers=False)
ax.legend(loc='upper right', ncol=1, title="", bbox_to_anchor=(1.0, 1.15), fontsize=14)
ax.set_xticks(np.arange(2), labels=["SG1", "SG2"], fontsize=14)
ax.set_yticks([-2.0, 0.0, 2.0])
ax.set_ylim(None, 5.25)
ax.set_ylabel("Mesulam", fontsize=14)
ax.set_xlabel("")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join("diffusion_neuromaps", "plots", "figs", "gradient_mesulam.png"), bbox_inches="tight", dpi=400, pad_inches=0.05)
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(3, 4), gridspec_kw={"hspace": 0.0, "wspace": 0.0})
ax = sns.boxplot(ax=ax, data=df_sa_axis, x="Metric", y="Value", hue="Region", order=["SG1", "SG2"], hue_order=["association", "sensorimotor"], palette=sa_pal, showfliers=False)
ax.legend(loc='upper right', ncol=1, title="", bbox_to_anchor=(1.0, 1.15), fontsize=14)
ax.set_xticks(np.arange(2), labels=["SG1", "SG2"], fontsize=14)
ax.set_yticks([-2.0, 0.0, 2.0])
ax.set_ylim(None, 5.25)
ax.set_ylabel("SA Axis", fontsize=14)
ax.set_xlabel("")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join("diffusion_neuromaps", "plots", "figs", "gradient_sa_axis.png"), bbox_inches="tight", dpi=400, pad_inches=0.05)
plt.close(fig)



