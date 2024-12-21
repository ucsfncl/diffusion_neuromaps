a



nnot_file = neuromaps.datasets.fetch_annotation(desc="ucbj", space="MNI152")
annot_parc = parcellator.fit_transform(annot_file, "MNI152", True).astype(float).ravel()
annot_parc = zscore(annot_parc)
df = pd.DataFrame(annot_parc, columns=["SV2A"], index=glasser_df.index).astype(float)
df.to_csv(os.path.join(metric_path, "glasser", "SV2A_glasser_ave.csv"), index=True, header=True)

annot_file = neuromaps.datasets.fetch_annotation(desc="ly2795050", space="MNI152")
annot_parc = parcellator.fit_transform(annot_file, "MNI152", True).astype(float).ravel()
annot_parc = zscore(annot_parc)
df = pd.DataFrame(annot_parc, columns=["KOR"], index=glasser_df.index).astype(float)
df.to_csv(os.path.join(metric_path, "glasser", "KOR_glasser_ave.csv"), index=True, header=True)


pet_path = os.path.join("hansen_receptors/data/PET_nifti_images")
pet_files = os.listdir(pet_path)

pet_parcellated= {}
for file in pet_files:
    pet_parcellation = parcellator.fit_transform(os.path.join(pet_path, file), "MNI152", True).astype(float).ravel()
    name = file.split('.')[0]
    pet_parcellated[name] = pet_parcellation
    # print(name)
    # print(pet_parcellation)
    # np.savetxt(os.path.join(metric_path, "pet_maps", f"{name}.csv"), pet_parcellation, delimiter=',')


pet_names = ["5HT1a_way_hc36_savli", "5HT1b_p943_hc22_savli", "5HT1b_p943_hc65_gallezot", "5HT2a_cimbi_hc29_beliveau",
             "5HT4_sb20_hc59_beliveau", "5HT6_gsk_hc30_radhakrishnan", "5HTT_dasb_hc100_beliveau", "A4B2_flubatine_hc30_hillmer",
             "CB1_omar_hc77_normandin", "D1_SCH23390_hc13_kaller", "D2_flb457_hc37_smith", "D2_flb457_hc55_sandiego",
             "DAT_fpcit_hc174_dukart_spect", "GABAa-bz_flumazenil_hc16_norgaard", "H3_cban_hc8_gallezot", "M1_lsn_hc24_naganawa",
             "mGluR5_abp_hc22_rosaneto", "mGluR5_abp_hc28_dubois", "mGluR5_abp_hc73_smart", "MU_carfentanil_hc204_kantonen",
             "NAT_MRB_hc77_ding", "NMDA_ge179_hc29_galovic", "VAChT_feobv_hc4_tuominen",
             "VAChT_feobv_hc5_bedard_sum", "VAChT_feobv_hc18_aghourian_sum"]



# combine all the receptors (including repeats)
r = np.zeros([360, len(pet_names)])
for i in range(len(pet_names)):
    r[:, i] = pet_parcellated[pet_names[i]]

receptor_names =["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                 "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                 "MOR", "NET", "NMDA", "VAChT"]

# make final region x receptor matrix

receptor_data = np.zeros([360, len(receptor_names)])
receptor_data[:, 0] = r[:, 0]
receptor_data[:, 2:9] = r[:, 3:10]
receptor_data[:, 10:14] = r[:, 12:16]
receptor_data[:, 15:18] = r[:, 19:22]

# weighted average of 5HT1B p943
receptor_data[:, 1] = (zscore(r[:, 1])*22 + zscore(r[:, 2])*65) / (22+65)

# weighted average of D2 flb457
receptor_data[:, 9] = (zscore(r[:, 10])*37 + zscore(r[:, 11])*55) / (37+55)

# weighted average of mGluR5 ABP688
receptor_data[:, 14] = (zscore(r[:, 16])*22 + zscore(r[:, 17])*28 + zscore(r[:, 18])*73) / (22+28+73)

# weighted average of VAChT FEOBV
receptor_data[:, 18] = (zscore(r[:, 22])*4 + zscore(r[:, 23]) + zscore(r[:, 24])) / \
                       (4+5+18)

for i in range(receptor_data.shape[1]):
    receptor_data[:, i] = zscore(receptor_data[:, i])
    df = pd.DataFrame(receptor_data[:, i], columns=[receptor_names[i]], index=glasser_df.index).astype(float)
    df.to_csv(os.path.join(metric_path, "glasser", f"{receptor_names[i]}_glasser_ave.csv"), index=True, header=True)

# df = pd.DataFrame(receptor_data, columns=receptor_names, index=glasser_df.index)

# print(df.head())

# df.to_csv(os.path.join(metric_path, "glasser", "receptors_glasser_ave.csv"), index=True, header=True)
