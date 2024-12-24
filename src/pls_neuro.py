import os
import numpy as np
import pandas as pd
import pyls
from tools import make_glasser_nulls, plot_glasser_map
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


###  1. Load the structural metrics and functional maps
metric_path = "diffusion_neuromaps/data"
glasser_df = pd.read_csv("diffusion_neuromaps/atlases/glasser_parc.csv", index_col=0)
metrics = [
    "dki-fa", "dki-ad", "dki-rd", "dki-md", "mk", "ak", "rk", "kfa", "mkt",
    "icvf", "odi", "isovf", "msd", "qiv", "rtop", "rtap", "rtpp", "thick", "myl"
    ]

dki_dti_metrics = ["dki-fa", "dki-ad", "dki-rd", "dki-md"]
dki_metrics = ["mk", "ak", "rk", "kfa", "mkt"]
noddi_metrics = ["icvf", "odi", "isovf"]
mapmri_metrics = ["msd", "qiv", "rtop", "rtap", "rtpp"]
struct_metrics = ["thick", "myl"]

struct_df = pd.DataFrame(index=glasser_df.index, columns=metrics, dtype=float)
for metric in metrics:
    hcp_df = pd.read_csv(os.path.join(metric_path, "glasser", "ave", f"{metric}_glasser_ave.csv"), index_col=0)
    struct_df[metric] = hcp_df


cognitive_terms = [
    "action",
    "eating",
    "insight",
    "naming",
    "semantic memory",
    "adaptation",
    "efficiency",
    "integration",
    "navigation",
    "sentence comprehension",
    "addiction",
    "effort",
    "intelligence",
    "object recognition",
    "skill",
    "anticipation",
    "emotion",
    "intention",
    "pain",
    "sleep",
    "anxiety",
    "emotion regulation",
    "interference",
    "perception",
    "social cognition",
    "arousal",
    "empathy",
    "judgment",
    "planning",
    "spatial attention",
    "association",
    "encoding",
    "knowledge",
    "priming",
    "speech perception",
    "attention",
    "episodic memory",
    "language",
    "psychosis",
    "speech production",
    "autobiographical memory",
    "expectancy",
    "language comprehension",
    "reading",
    "strategy",
    "balance",
    "expertise",
    "learning",
    "reasoning",
    "strength",
    "belief",
    "extinction",
    "listening",
    "recall",
    "stress",
    "categorization",
    "face recognition",
    "localization",
    "recognition",
    "sustained attention",
    "cognitive control",
    "facial expression",
    "loss",
    "rehearsal",
    "task difficulty",
    "communication",
    "familiarity",
    "maintenance",
    "reinforcement learning",
    "thought",
    "competition",
    "fear",
    "manipulation",
    "response inhibition",
    "uncertainty",
    "concept",
    "fixation",
    "meaning",
    "response selection",
    "updating",
    "consciousness",
    "focus",
    "memory",
    "retention",
    "utility",
    "consolidation",
    "gaze",
    "memory retrieval",
    "retrieval",
    "valence",
    "context",
    "goal",
    "mental imagery",
    "reward anticipation",
    "verbal fluency",
    "coordination",
    "hyperactivity",
    "monitoring",
    "rhythm",
    "visual attention",
    "decision",
    "imagery",
    "mood",
    "risk",
    "visual perception",
    "decision making",
    "impulsivity",
    "morphology",
    "rule",
    "word recognition",
    "detection",
    "induction",
    "motor control",
    "salience",
    "working memory",
    "discrimination",
    "inference",
    "movement",
    "search",
    "distraction",
    "inhibition",
    "multisensory",
    "selective attention"
]

cog_df = pd.DataFrame(index=glasser_df.index, columns=cognitive_terms, dtype=float)
for name in cognitive_terms:
    cog_df[name] = pd.read_csv(os.path.join(metric_path, "neuroquery", f"{name}_glasser_ave.csv"), index_col=0)

###  2. Run the PLS analysis
nspins = 9999 
gen = make_glasser_nulls(nspins)
spins = np.hstack(gen.randomize(np.arange(180), np.arange(180, 360))).T

X = scipy.stats.zscore(struct_df.values, axis=0)
Y = scipy.stats.zscore(cog_df.values, axis=0)

pls_result = pyls.behavioral_pls(X, Y, n_boot=nspins, n_perm=nspins, permsamples=spins, test_split=0, seed=0)

###  3. Save and plot the scores
df = pd.DataFrame(pls_result["x_scores"][:, 0], index=glasser_df.index, columns=["struct_score"])
df.index.name = "region"
df.to_csv("diffusion_neuromaps/data/pls/neuro_pls_struct_score_glasser_ave.csv")

df = pd.DataFrame(pls_result["y_scores"][:, 0], index=glasser_df.index, columns=["pet_score"])
df.index.name = "region"
df.to_csv("diffusion_neuromaps/data/pls/neuro_pls_cog_score_glasser_ave.csv")

plot_glasser_map(pls_result["x_scores"][:, 0], size=(1600, 400), cmap="RdBu_r", label_text=None, color_bar=False, color_range="sym",
nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False, 
screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", "neuro_pls_struct_score.png"))

plot_glasser_map(pls_result["y_scores"][:, 0], size=(1600, 400), cmap="RdBu_r", label_text=None, color_bar=False, color_range="sym",
nan_color=(255, 255, 255, 1), zoom=1.25, transparent_bg=False, interactive=False,
screenshot=True, filename=os.path.join("diffusion_neuromaps", "plots", "glasser", "neuro_pls_cog_score.png"))


###  4. Plot and save the structural loadings with confidence intervals
xload = pyls.behavioral_pls(Y, X, n_boot=nspins, n_perm=0, test_split=0, permsamples=spins, seed=0)
df = pd.DataFrame(xload["y_loadings"][:, 0], index=metrics, columns=["struct_loading"])
df.index.name = "metric"
df.to_csv("diffusion_neuromaps/data/pls/neuro_pls_struct_loadings.csv")

err = (xload["bootres"]["y_loadings_ci"][:, 0, 1]
      - xload["bootres"]["y_loadings_ci"][:, 0, 0]) / 2
sorted_idx = np.argsort(xload["y_loadings"][:, 0])

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(1, 1, figsize=(3, 6), gridspec_kw={"wspace": 0.0, "hspace": 0.0})
ax.barh(range(len(metrics)), xload["y_loadings"][sorted_idx, 0], xerr=err[sorted_idx], color=["red" if xload["y_loadings"][sorted_idx[i], 0] > 0 else "blue" for i in range(len(sorted_idx))])
ax.set_yticks(range(len(metrics)), labels=[metrics[i].upper() for i in sorted_idx], fontsize=14)
ax.set_xlabel("Structural Loadings", fontsize=14)
fig.tight_layout()
plt.savefig("diffusion_neuromaps/plots/figs/neuro_pls_struct_load.png", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

###  5. Plot the cognitive loadings with confidence intervals
df = pd.DataFrame(pls_result["y_loadings"][:, 0], index=cognitive_terms, columns=["cog_loading"])
df.index.name = "cognitive term"
df.to_csv("diffusion_neuromaps/data/pls/neuro_pls_cog_loadings.csv")

err = (pls_result["bootres"]["y_loadings_ci"][:, 0, 1]
      - pls_result["bootres"]["y_loadings_ci"][:, 0, 0]) / 2
sorted_idx = np.argsort(pls_result["y_loadings"][:, 0])
top_20 = sorted_idx[-20:]
bottom_20 = sorted_idx[:20]

fig, ax = plt.subplots(1, 1, figsize=(6, 6), gridspec_kw={"wspace": 0.0, "hspace": 0.0})
ax.barh(np.arange(20), pls_result["y_loadings"][top_20, 0], xerr=err[top_20], color="red")
ax.set_yticks(range(20), labels=[cognitive_terms[i] for i in bottom_20], fontsize=14)
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(20), labels=[cognitive_terms[i] for i in top_20], fontsize=14)
ax2.barh(np.arange(20), pls_result["y_loadings"][bottom_20, 0], xerr=err[bottom_20], color="blue")
ax.set_xlabel("Cognitive Loadings", fontsize=14)
fig.tight_layout()
fig.savefig("diffusion_neuromaps/plots/figs/neuro_pls_cog_load.png", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

###  6. Find the explained variance of the first latent variable and plot the explained variance of each latent variable
cv = pls_result["singvals"]**2 / np.sum(pls_result["singvals"]**2)
p = pls_result["permres"]["pvals"][0]

print(f"The first latent variable explains {cv[0]*100:.2f}% of the covariance, with a p-value of {p:.5f}")

plt.figure(figsize=(4, 4))
plt.scatter(range(len(metrics)), cv*100, s=80)
plt.ylabel("percent covariance accounted for")
plt.xlabel("latent variable")
plt.title('PLS' + str(0) + ' cov exp = ' + str(cv[0])[:5]
          + ', pspin = ' + str(p)[:5])
plt.tight_layout()
plt.savefig("diffusion_neuromaps/plots/figs/neuro_pls_explained_var.png", dpi=400, bbox_inches="tight", pad_inches=0.05)