# python3 exe_train_val_test_split.py --dataset='tmp.txt' --trainset='trainset_snp.txt' --valset='valset_snp.txt' --testset='testset_snp.txt' --random_seed=1337 --sampling='snp'
# python3 exe_train_val_test_split.py --dataset='tmp.txt' --trainset='trainset_locus.txt' --valset='valset_locus.txt' --testset='testset_locus.txt' --random_seed=1337 --sampling='locus'

import math
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', help='dataset file name')
    argparser.add_argument('--trainset', help='trainset file name')
    argparser.add_argument('--valset', help='valset file name')
    argparser.add_argument('--testset', help='testset file name')
    argparser.add_argument('--random_seed', help='random seed')
    argparser.add_argument('--sampling', choices=['snp', 'locus'], help='sampling method: snp level sampling or locus sampling')
    args = argparser.parse_args()

    filename_dataset = args.dataset
    filename_trainset = args.trainset
    filename_valset = args.valset
    filename_testset = args.testset
    random_seed = int(args.random_seed)
    sampling = args.sampling

    dataset = pd.read_csv(filename_dataset, sep="\t")

    if sampling == 'snp':
        X = dataset.drop('label', axis=1)
        y = dataset['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_seed, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_seed, stratify=y_train)

        trainset = pd.concat([X_train, y_train], axis=1)
        valset = pd.concat([X_val, y_val], axis=1)
        testset = pd.concat([X_test, y_test], axis=1)

    elif sampling == 'locus':
        np.random.seed(random_seed)
        rsid_list = np.random.permutation(dataset[dataset['label']==1]['rsid'].unique())

        idx_train_start = 0
        idx_train_end = math.ceil(len(rsid_list)*0.6)
        idx_val_start = idx_train_end
        idx_val_end = math.ceil(len(rsid_list)*0.8)
        idx_test_start = idx_val_end
        idx_test_end = len(rsid_list)
        rsid_train = rsid_list[idx_train_start : idx_train_end]
        rsid_val = rsid_list[idx_val_start : idx_val_end]
        rsid_test = rsid_list[idx_test_start : idx_test_end]

        trainset = dataset[dataset['query_snp_rsid'].isin(rsid_train)]
        valset = dataset[dataset['query_snp_rsid'].isin(rsid_val)]
        testset = dataset[dataset['query_snp_rsid'].isin(rsid_test)]

    trainset.to_csv(filename_trainset, sep="\t", index=False)
    print("[INFO] trainset saved to {}".format(filename_trainset))
    valset.to_csv(filename_valset, sep="\t", index=False)
    print("[INFO] valset saved to {}".format(filename_valset))
    testset.to_csv(filename_testset, sep="\t", index=False)
    print("[INFO] testset saved to {}".format(filename_testset))
