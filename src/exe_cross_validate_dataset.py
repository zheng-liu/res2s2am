import os
import pandas as pd

seq_length_list = [200, 500, 1000, 2000, 4000]
random_seed_list = [1337, 1338, 1339, 1340, 1341]
sampling = ['snp', 'locus']

for sam in sampling:
    for seq_length in seq_length_list:
        for random_seed in random_seed_list:
            filename_dataset = 'snap_data_w_annot_{}_seq_reg.txt'.format(seq_length)
            filename_trainset = 'trainset_{}_{}_fold_{}.txt'.format(sam, str(seq_length), str(random_seed))
            filename_valset = 'valset_{}_{}_fold_{}.txt'.format(sam, str(seq_length), str(random_seed))
            filename_testset = 'testset_{}_{}_fold_{}.txt'.format(sam, str(seq_length), str(random_seed))
            command = 'python3 exe_train_val_test_split.py --dataset={} --trainset={} --valset={} --testset={} --random_seed={} --sampling={}'.format(filename_dataset, filename_trainset, filename_valset, filename_testset, str(random_seed), sam)
            os.system(command)
            print('[INFO] generate {}, {}, {}!'.format(filename_trainset, filename_valset, filename_testset))
