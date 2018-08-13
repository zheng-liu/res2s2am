#!/bin/sh

# test regulomedb score
python3 test_regulomedb.py --test_kwargs='test_regulomedb_1000_fold_1337.p' --testset='testset_snp_1000_fold_1337.txt' --sampling='snp' --output='out_test_regulomedb_1000_fold_1337.txt' --roc='roc_test_regulomedb_1000_fold_1337.txt' --prc='prc_test_regulomedb_1000_fold_1337.txt'
wait
python3 test_regulomedb.py --test_kwargs='test_regulomedb_1000_fold_1338.p' --testset='testset_snp_1000_fold_1338.txt' --sampling='snp' --output='out_test_regulomedb_1000_fold_1338.txt' --roc='roc_test_regulomedb_1000_fold_1338.txt' --prc='prc_test_regulomedb_1000_fold_1338.txt'
wait
python3 test_regulomedb.py --test_kwargs='test_regulomedb_1000_fold_1339.p' --testset='testset_snp_1000_fold_1339.txt' --sampling='snp' --output='out_test_regulomedb_1000_fold_1339.txt' --roc='roc_test_regulomedb_1000_fold_1339.txt' --prc='prc_test_regulomedb_1000_fold_1339.txt'
wait
python3 test_regulomedb.py --test_kwargs='test_regulomedb_1000_fold_1340.p' --testset='testset_snp_1000_fold_1340.txt' --sampling='snp' --output='out_test_regulomedb_1000_fold_1340.txt' --roc='roc_test_regulomedb_1000_fold_1340.txt' --prc='prc_test_regulomedb_1000_fold_1340.txt'
wait
python3 test_regulomedb.py --test_kwargs='test_regulomedb_1000_fold_1341.p' --testset='testset_snp_1000_fold_1341.txt' --sampling='snp' --output='out_test_regulomedb_1000_fold_1341.txt' --roc='roc_test_regulomedb_1000_fold_1341.txt' --prc='prc_test_regulomedb_1000_fold_1341.txt'
wait
