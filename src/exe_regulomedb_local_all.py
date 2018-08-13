import os
import pandas as pd

seq_length_list = [200, 500, 1000, 2000, 4000]

for seq_length in seq_length_list:
    filename_snap_data = 'snap_data_w_annot_{}_seq.txt'.format(seq_length)
    filename_reg_data = 'snap_data_w_annot_{}_seq_reg.txt'.format(seq_length)
    command = 'python3 exe_regulomedb_local.py {} {}'.format(filename_snap_data, filename_reg_data)
    os.system(command)
    print('[INFO] generate {}!'.format(filename_reg_data)
