# diffusion_neuromaps
Code for microstructural mapping of the human cortex using diffusion MRI
See preprint at https://www.biorxiv.org/content/10.1101/2024.09.27.615479v2

## atlases
[atlases](atlases/) folder contains Glasser, DK, and various other maps/parcellations and coordinates

## data
[data](data/) folder contains all relevant data needed to run almost all the code. We do not provide subjectwise fsLR32k data due to the large data requirements but do provide code for generating the data once you have downloaded the HCP dataset.
- [dk](data/dk/) has all the average, scn, and subject data parcellated by the DK atlas
- [fslr_32k](data/fslr_32k/) has the average maps as fsLR32k surfaces
- [glasser](data/glasser/) has all the average, laterality index, intersubject coefficient of variation (CoV), test-retest CoV, and test-retest intraclass correlation coefficient (ICC)
- [neuromaps](data/neuromaps/) has other Glasser and DK neuromaps
- [pet](data/pet) has PET receptor/transporter densities parcellated by Glasser and DK atlases

## plots
[plots](plots) folder contains all plots as .png files
- [figs](plots/figs/) has figures showing the results of the analysis
- [glasser](plots/glasser/) has figures showing various glasser maps

## src
[src](src) folder contains all code used for the analysis.
- [dk_parc](src/dk_parc.py) generates the DK parcellated data
- [factor_analysis](src/factor_analysis.py) performs factor analysis on the microstructural metrics
- [get_spherical_pts](src/get_spherical_pts.py) obtains the spherical coordinates used for spin-permutation testing
- [glasser_parc](src/glasser_parc.py) generates the Glasser parcellated data
- [meg_corr](src/meg_corr.py) performs MEG univariate and multivariate correlation analysis
- [pet_corr](src/pet_corr.py) performs PET univariate and multivariate correlation analysis
- [plot_glasser_corr](src/plot_glasser_corr.py) performs similarity analysis among the microstructural metrics
- [scn_connectivity](src/scn_connectivity.py) perfoms SCN connectivity analysis
- [scn_corr](src/scn_corr.py) correlates SCNs with FC & SC and gene & PET similarity and microstructure with their degree centrality
- [scn](src/scn.py) generates structural covariance networks (SCNs) for each microstructural metric
- [tools](src/tools.py) has various utility functions used throughout the other scripts

Some sections of code are heavily borrowed from other repositories:
See https://github.com/netneurolab/hansen_receptors/tree/main