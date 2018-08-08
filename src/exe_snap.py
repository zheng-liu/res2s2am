import os
import sys
import snap_v5
import pandas as pd
import numpy as np
import time
import pickle

positive_sub_file = str(sys.argv[1])

start_time = time.time()
positive_sub = pd.read_csv(positive_sub_file, sep="\t")
positive_rsid_all = list(set(positive_sub["SNPid(dbSNP134)"]))
pickle.dump(positive_rsid_all, open('rsid.p', 'wb'))
print('[INFO] rsid pickled to rsid.p')
print(len(positive_rsid_all))

# SNAP Proxy
search_kwargs = {}
search_kwargs['ldThresh'] = 0.80
search_kwargs['ldPop'] = ['AFR', 'AMR', 'ASN', 'EUR']
search_kwargs['epi'] = 'vanilla'
search_kwargs['cons'] = 'both'
search_kwargs['genetypes'] = 'both'
search_kwargs['trunc'] = 1000
search_kwargs['oligo'] = 1000
search_kwargs['output'] = 'text'
search_kwargs['maf_thresh'] = 0.05
search_kwargs['max_attempt'] = 1

snap_data, partial_list, fail_list = snap_v5.fast_get_locus_map(positive_rsid_all, search_kwargs, True)
snap_data.to_csv('snap_data_raw.txt', sep='\t', index=False)
print('[INFO] raw snap_data is saved to snap_data_raw.txt!')

# incompletely extracted SNPs from HaploReg
with open('partial_list.txt', 'w') as pl:
    for rsid in partial_list:
        pl.write('{}\n'.format(rsid))
print('[INFO] partial_list.txt saved!')

# failed to extract SNPs from HaploReg
with open('fail_list.txt', 'w') as fl:
    for rsid in fail_list:
        fl.write('{}\n'.format(rsid))
print('[INFO] fail_list.txt saved!')

# filter proxy by removing snps in GRASP
grasp_rsid = list(set(pd.read_csv("grasp_sub.txt", sep="\t")["SNPid(dbSNP134)"]))
snap_data = snap_data[(~snap_data["rsID"].isin(grasp_rsid)) | (snap_data['is_query_snp']==1)] # select non-gwas SNPs (SNPs out of GRASP database)
snap_data.drop_duplicates(subset=['rsID', 'query_snp_rsid'], keep='first', inplace=True) # remove duplicates among populations (the values among populations are identical)

snap_data.sort_values(['query_snp_rsid', 'pos_hg38', 'pop'], ascending=[True, True, True], inplace=True)
snap_data.reset_index(drop=True, inplace=True)
snap_data.drop(columns='pop', inplace=True)
snap_data.replace({'.': np.nan}, inplace=True)
snap_data.to_csv("snap_data_notin_grasp.txt", sep="\t", index=False)
print('[INFO] snap_data_notin_grasp.txt saved!')
print('[INFO] time={}s'.format(time.time() - start_time))
