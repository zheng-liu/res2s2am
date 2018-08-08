import pandas as pd
import annotate
import genome_browser

snap_data_notin_grasp = pd.read_csv('snap_data_notin_grasp.txt', sep='\t').rename(columns={'pos_hg38': 'pos', 'rsID': 'rsid'})

# remove protein-coding SNPs
annotate_ob = annotate.annotate(snap_data_notin_grasp, 'hg38')
snap_data_w_annot = annotate_ob.fast_annotate_snp_list()
snap_data_w_annot = snap_data_w_annot.loc[snap_data_w_annot['annotation'] != 'pcexon']
snap_data_w_annot = snap_data_w_annot.sort_values(['query_snp_rsid', 'rsid']).reset_index(drop=True)
snap_data_w_annot.to_csv('snap_data_w_annot.txt', sep='\t', index=False)
print('[INFO] snap_data_w_annot.txt saved!')
