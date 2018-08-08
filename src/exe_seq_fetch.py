import sys
import pandas as pd
import genome_browser

# fetch sequence chromosome by chromosome
chrom = str(sys.argv[1])
length = int(sys.argv[2])
filename = 'snap_data_w_annot_{}.txt'.format(chrom)

snap_data_w_annot_chr = pd.read_csv(filename, sep='\t')
snap_data_w_annot_chr_seq = genome_browser.fast_fetch_seq(snap_data_w_annot_chr, chrom, 'hg38', length)
snap_data_w_annot_chr_seq.to_csv('snap_data_w_annot_{}_{}_seq.txt'.format(chrom, str(length)), sep='\t', index=False)
print('[INFO] snap_data_w_annot_{}_{}_seq.txt'.format(chrom, str(length)))
