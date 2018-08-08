import pandas as pd

# split by chromsome for sequence expansion
__chrom_set = ['chr'+str(i) for i in range(1, 23)] + ['chrX', 'chrY']
snap_data_w_annot = pd.read_csv('snap_data_w_annot.txt', sep='\t')
for c in __chrom_set:
    filename = 'snap_data_w_annot_{}.txt'.format(c)
    snap_data_w_annot[snap_data_w_annot['chr'] == c].to_csv(filename, sep='\t', index=False)
    print('[INFO] {} saved to {}'.format(c, filename))
