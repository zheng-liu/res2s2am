# python3 test_resnet_2s2a_metadata.py --device=1 --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1337.p' --testset='testset_snp_1000_fold_1337.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1337.pt' --pred='pred_test_resnet_2s2a_metadata_1000_fold_1337.txt' --phenotype_dist='phenotype_dist_test_resnet_2s2a_metadata_1000_fold_1337.txt'
# python3 test_resnet_2s2a_metadata.py --device=1 --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1338.p' --testset='testset_snp_1000_fold_1338.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1338.pt' --pred='pred_test_resnet_2s2a_metadata_1000_fold_1338.txt' --phenotype_dist='phenotype_dist_test_resnet_2s2a_metadata_1000_fold_1338.txt'
# python3 test_resnet_2s2a_metadata.py --device=1 --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1339.p' --testset='testset_snp_1000_fold_1339.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1339.pt' --pred='pred_test_resnet_2s2a_metadata_1000_fold_1339.txt' --phenotype_dist='phenotype_dist_test_resnet_2s2a_metadata_1000_fold_1339.txt'
# python3 test_resnet_2s2a_metadata.py --device=1 --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1340.p' --testset='testset_snp_1000_fold_1340.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1340.pt' --pred='pred_test_resnet_2s2a_metadata_1000_fold_1340.txt' --phenotype_dist='phenotype_dist_test_resnet_2s2a_metadata_1000_fold_1340.txt'
# python3 test_resnet_2s2a_metadata.py --device=1 --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1341.p' --testset='testset_snp_1000_fold_1341.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1341.pt' --pred='pred_test_resnet_2s2a_metadata_1000_fold_1341.txt' --phenotype_dist='phenotype_dist_test_resnet_2s2a_metadata_1000_fold_1341.txt'

import argparse
import pickle
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn import metrics
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader_2 import GenomeDataset

def make_kwargs(filename_test_kwargs):
    # test_kwargs, resnet_2s2a_metadata
    test_kwargs_resnet_2s2a_metadata = {}
    test_kwargs_resnet_2s2a_metadata['random_seed'] = 1337
    test_kwargs_resnet_2s2a_metadata['cudnn'] = True
    test_kwargs_resnet_2s2a_metadata['batch_size'] = 256
    test_kwargs_resnet_2s2a_metadata['encode_mode'] = 'N_2_zero'
    test_kwargs_resnet_2s2a_metadata['shuffle'] = True
    test_kwargs_resnet_2s2a_metadata['num_workers'] = 8
    test_kwargs_resnet_2s2a_metadata['metadata_cols'] = ['AFR', 'AMR', 'ASN', 'EUR', 'GERP_cons', 'SiPhy_cons', 'reg_score_int']
    pickle.dump(test_kwargs_resnet_2s2a_metadata, open(filename_test_kwargs, 'wb'))
    print('[INFO] {} dumped!'.format(filename_test_kwargs))
    return test_kwargs_resnet_2s2a_metadata

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # tuning parameters
    argparser.add_argument('--test_kwargs', help='kwargs of network model testing.')
    argparser.add_argument('--testset', help='testset file name.')
    argparser.add_argument('--sampling', choices=['snp', 'locus'], help='Train, validation, testing sampling method.', required=True)
    argparser.add_argument('--model', help='model file name')
    argparser.add_argument('--pred', help='prediction file name')
    argparser.add_argument('--phenotype_dist', help='phenotype_distribution file name')
    argparser.add_argument('--device', help='device number.')
    args = argparser.parse_args()

    filename_test_kwargs = args.test_kwargs
    test_kwargs = make_kwargs(filename_test_kwargs)
    filename_testset = args.testset
    filename_model = args.model
    filename_pred = args.pred
    filename_phenotype_dist = args.phenotype_dist
    sampling = args.sampling
    device = int(args.device)

    cudnn = test_kwargs['cudnn']
    shuffle = test_kwargs['shuffle']
    random_seed = test_kwargs['random_seed']
    encode_mode = test_kwargs['encode_mode']
    batch_size = test_kwargs['batch_size']
    num_workers = test_kwargs['num_workers']
    metadata_cols = test_kwargs['metadata_cols']

    torch.backends.cudnn.fastest = cudnn
    torch.cuda.set_device(device) # assign gpu

    correct = 0.0
    tp = 0.0
    size = 0.0
    # map_location: map cuda gpu numbers (specially for transfering from different workstations)
    model = torch.load(filename_model, map_location=dict(('cuda:'+str(k), 'cuda:'+str(device)) for k in range(0, 100)))
    model.train(False)
    testset = GenomeDataset(filename=filename_testset, seq_name=['seq_ref_1', 'seq_ref_2', 'seq_alt_1', 'seq_alt_2'], encode_mode=encode_mode, metadata_cols=metadata_cols)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print('[INFO] testloader generated')

    labels_list = []
    rsids_list = []
    preds_list = []
    query_snp_rsids_list = []

    for data in tqdm(testloader):
        inputs1, inputs2, inputs3, inputs4, labels, rsids, query_snp_rsids, metadata = data

        if torch.cuda.is_available():
            inputs1, inputs2, inputs3, inputs4, labels, metadata = Variable(inputs1.cuda()), Variable(inputs2.cuda()), Variable(inputs3.cuda()), Variable(inputs4.cuda()), Variable(labels.cuda()), Variable(metadata.cuda())
        else:
            inputs1, inputs2, inputs3, inputs4, labels, metadata = Variable(inputs1), Variable(inputs2), Variable(inputs3), Variable(inputs4), Variable(labels), Variable(metadata)

        outputs = model(inputs1, inputs2, inputs3, inputs4, metadata)
        _, preds = torch.max(outputs.data, 1)

        rsids_list.extend(rsids)
        preds_list.extend(preds)
        labels_list.extend(labels)
        query_snp_rsids_list.extend(query_snp_rsids)

    pred_df = pd.DataFrame({'rsid': rsids_list, 'pred': preds_list, 'label': labels_list, 'query_snp_rsid': query_snp_rsids_list})
    # select query SNPs for ranking
    # //TODO check if query is correct
    pred_df = pred_df.query('rsid == query_snp_rsid')
    pred_df.to_csv(filename_pred, sep='\t')
    print('[INFO] {} saved!'.format(filename_pred))

    # calculate the trait/disease categories rank
    grasp_full = pd.read_csv('grasp_full.txt', sep='\t')
    phenotype_df = grasp_full[['SNPid(dbSNP134)', 'Phenotype', 'PaperPhenotypeCategories']].rename(columns={'SNPid(dbSNP134)': 'rsid'})
    pred_df = pd.merge(pred_df, phenotype_df, how='left', on='rsid')

    phenotype_hit = {}
    phenotype_sum={}
    for index, row in pred_df.iterrows():
        phenoCat = row['PaperPhenotypeCategories'].split(';')
        for ph in phenoCat:
            if ph in phenotype_sum:
                phenotype_sum[ph] += 1
            else:
                phenotype_sum[ph] = 1

            if ph in phenotype_hit:
                if row['pred'] == row['label']:
                    phenotype_hit[ph] += 1
            else:
                if row['pred'] == row['label']:
                    phenotype_hit[ph] = 1
                else:
                    phenotype_hit[ph] = 0

    # phenotype_dist_sorted = sorted(phenotype_dist.items(), key=lambda item: item[1], reverse=True)
    # phenotype_dist_sorted = pd.DataFrame(phenotype_dist_sorted, columns=['phenotype', 'hit'])
    # phenotype_dist_sorted.to_csv(filename_phenotype_dist, sep='\t')
    # print('[INFO] {} saved'.format(filename_phenotype_dist))

    phenotype_dist_df = pd.DataFrame({'hit': phenotype_hit, 'sum': phenotype_sum})
    phenotype_dist_df.to_csv(filename_phenotype_dist, sep='\t')
    # phenotype_dist_df['ratio'] = phenotype_dist_df.hit.div(phenotype_dist_df.sum, axis=0)
    print('[INFO] {} saved'.format(filename_phenotype_dist))
