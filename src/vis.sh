#!/bin/sh

# rank phenotypes by prediction accuracies from resnet_2s2a_metadata model
python3 test_resnet_2s2a_metadata.py --device=? --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1337.p' --testset='testset_snp_1000_fold_1337.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1337.pt' --pred='pred_test_resnet_2s2a_metadata_1000_fold_1337.txt' --phenotype_dist='phenotype_dist_test_resnet_2s2a_metadata_1000_fold_1337.txt'
wait
python3 test_resnet_2s2a_metadata.py --device=? --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1338.p' --testset='testset_snp_1000_fold_1338.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1338.pt' --pred='pred_test_resnet_2s2a_metadata_1000_fold_1338.txt' --phenotype_dist='phenotype_dist_test_resnet_2s2a_metadata_1000_fold_1338.txt'
wait
python3 test_resnet_2s2a_metadata.py --device=? --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1339.p' --testset='testset_snp_1000_fold_1339.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1339.pt' --pred='pred_test_resnet_2s2a_metadata_1000_fold_1339.txt' --phenotype_dist='phenotype_dist_test_resnet_2s2a_metadata_1000_fold_1339.txt'
wait
python3 test_resnet_2s2a_metadata.py --device=? --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1340.p' --testset='testset_snp_1000_fold_1340.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1340.pt' --pred='pred_test_resnet_2s2a_metadata_1000_fold_1340.txt' --phenotype_dist='phenotype_dist_test_resnet_2s2a_metadata_1000_fold_1340.txt'
wait
python3 test_resnet_2s2a_metadata.py --device=? --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1341.p' --testset='testset_snp_1000_fold_1341.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1341.pt' --pred='pred_test_resnet_2s2a_metadata_1000_fold_1341.txt' --phenotype_dist='phenotype_dist_test_resnet_2s2a_metadata_1000_fold_1341.txt'
wait

# visualize
python3 exe_vis_2.py