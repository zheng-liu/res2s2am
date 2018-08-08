import os
import sys
import pandas as pd

length = str(sys.argv[1])
filename = 'snap_data_w_annot_{}_seq.txt'.format(length)

# remove the isolated control SNPs without at least one query SNP
snap_data_w_annot_length_mode_seq = pd.read_csv(filename, sep='\t')
rsid_list_missing = list(set(snap_data_w_annot_length_mode_seq['query_snp_rsid']) - set(snap_data_w_annot_length_mode_seq['rsid']))
if len(rsid_list_missing) > 0:
    snap_data_w_annot_length_mode_seq = snap_data_w_annot_length_mode_seq.loc[~snap_data_w_annot_length_mode_seq['query_snp_rsid'].isin(rsid_list_missing)].sort_values(['rsid', 'query_snp_rsid'])
snap_data_w_annot_length_mode_seq.to_csv(filename, sep='\t', index=False)
print('[INFO] cleaned snap_data_w_annot_{}_seq.txt saved!'.format(length))
