#!/bin/sh

# train cnn_1s
python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_1000_fold_1337.p' --train_kwargs='train_kwargs_cnn_1s_1000_fold_1337.p' --trainset='trainset_snp_1000_fold_1337.txt' --valset='valset_snp_1000_fold_1337.txt' --sampling='snp' --output='out_train_cnn_1s_1000_fold_1337.txt' --log='log_train_cnn_1s_1000_fold_1337.txt' --model='model_cnn_1s_1000_fold_1337.pt'
wait
python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_1000_fold_1338.p' --train_kwargs='train_kwargs_cnn_1s_1000_fold_1338.p' --trainset='trainset_snp_1000_fold_1338.txt' --valset='valset_snp_1000_fold_1338.txt' --sampling='snp' --output='out_train_cnn_1s_1000_fold_1338.txt' --log='log_train_cnn_1s_1000_fold_1338.txt' --model='model_cnn_1s_1000_fold_1338.pt'
wait
python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_1000_fold_1339.p' --train_kwargs='train_kwargs_cnn_1s_1000_fold_1339.p' --trainset='trainset_snp_1000_fold_1339.txt' --valset='valset_snp_1000_fold_1339.txt' --sampling='snp' --output='out_train_cnn_1s_1000_fold_1339.txt' --log='log_train_cnn_1s_1000_fold_1339.txt' --model='model_cnn_1s_1000_fold_1339.pt'
wait
python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_1000_fold_1340.p' --train_kwargs='train_kwargs_cnn_1s_1000_fold_1340.p' --trainset='trainset_snp_1000_fold_1340.txt' --valset='valset_snp_1000_fold_1340.txt' --sampling='snp' --output='out_train_cnn_1s_1000_fold_1340.txt' --log='log_train_cnn_1s_1000_fold_1340.txt' --model='model_cnn_1s_1000_fold_1340.pt'
wait
python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_1000_fold_1341.p' --train_kwargs='train_kwargs_cnn_1s_1000_fold_1341.p' --trainset='trainset_snp_1000_fold_1341.txt' --valset='valset_snp_1000_fold_1341.txt' --sampling='snp' --output='out_train_cnn_1s_1000_fold_1341.txt' --log='log_train_cnn_1s_1000_fold_1341.txt' --model='model_cnn_1s_1000_fold_1341.pt'
wait