import os
import sys
import pandas as pd

# add phenotypes

length = str(sys.argv[1])

snap_data_filename = 'snap_data_w_annot_{}_seq.txt'.format(length)
snap_data_phenotype_filename = 'snap_data_w_annot_{}_seq_phenotype.txt'.format(length)
grasp_full_filename = 'grasp_full.txt'

if os.path.exists(snap_data_filename):
    snap_data = pd.read_csv(snap_data_filename, sep='\t')
else:
    print('[ERROR] {} not exists!'.format(snap_data_filename))
    sys.exit(0)

if os.path.exists(grasp_full_filename):
    grasp_full = pd.read_csv(grasp_full_filename, sep='\t')
else:
    print('[ERROR] {} not exists!'.format(grasp_full_filename))
    sys.exit(0)

snap_data_phenotype = pd.merge(snap_data, grasp_full[['SNPid(dbSNP134)', 'Phenotype', 'PaperPhenotypeDescription', 'PaperPhenotypeCategories']].rename(columns={'SNPid(dbSNP134)': 'rsid'}), on='rsid', how='left')
snap_data_phenotype.to_csv(snap_data_phenotype_filename, sep='\t', index=False)
print('[INFO] phenotype added to {}!'.format(snap_data_phenotype_filename))
