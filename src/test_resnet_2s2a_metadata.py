# python3 test_resnet_2s2a_metadata.py --device=? --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1337.p' --testset='testset_snp_1000_fold_1337.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1337.pt' --output='out_test_resnet_2s2a_metadata_1000_fold_1337.txt' --roc='roc_test_resnet_2s2a_metadata_1000_fold_1337.txt' --prc='prc_test_resnet_2s2a_metadata_1000_fold_1337.txt'
# python3 test_resnet_2s2a_metadata.py --device=? --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1338.p' --testset='testset_snp_1000_fold_1338.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1338.pt' --output='out_test_resnet_2s2a_metadata_1000_fold_1338.txt' --roc='roc_test_resnet_2s2a_metadata_1000_fold_1338.txt' --prc='prc_test_resnet_2s2a_metadata_1000_fold_1338.txt'
# python3 test_resnet_2s2a_metadata.py --device=? --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1339.p' --testset='testset_snp_1000_fold_1339.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1339.pt' --output='out_test_resnet_2s2a_metadata_1000_fold_1339.txt' --roc='roc_test_resnet_2s2a_metadata_1000_fold_1339.txt' --prc='prc_test_resnet_2s2a_metadata_1000_fold_1339.txt'
# python3 test_resnet_2s2a_metadata.py --device=? --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1340.p' --testset='testset_snp_1000_fold_1340.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1340.pt' --output='out_test_resnet_2s2a_metadata_1000_fold_1340.txt' --roc='roc_test_resnet_2s2a_metadata_1000_fold_1340.txt' --prc='prc_test_resnet_2s2a_metadata_1000_fold_1340.txt'
# python3 test_resnet_2s2a_metadata.py --device=? --test_kwargs='test_kwargs_resnet_2s2a_metadata_1000_fold_1341.p' --testset='testset_snp_1000_fold_1341.txt' --sampling='snp' --model='model_resnet_2s2a_metadata_1000_fold_1341.pt' --output='out_test_resnet_2s2a_metadata_1000_fold_1341.txt' --roc='roc_test_resnet_2s2a_metadata_1000_fold_1341.txt' --prc='prc_test_resnet_2s2a_metadata_1000_fold_1341.txt'

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
    argparser.add_argument('--output', help='output file name')
    argparser.add_argument('--roc', help='roc file name')
    argparser.add_argument('--prc', help='prc file name')
    argparser.add_argument('--device', help='device number.')
    args = argparser.parse_args()

    filename_test_kwargs = args.test_kwargs
    test_kwargs = make_kwargs(filename_test_kwargs)
    filename_testset = args.testset
    filename_model = args.model
    filename_output = args.output
    filename_roc = args.roc
    filename_prc = args.prc
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
    #model = torch.load(filename_model)
    model.train(False)
    testset = GenomeDataset(filename=filename_testset, seq_name=['seq_ref_1', 'seq_ref_2', 'seq_alt_1', 'seq_alt_2'], encode_mode=encode_mode, metadata_cols=metadata_cols)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print('[INFO] testloader generated')

    probs_list = []
    labels_list = []
    rsids_list = []
    query_snp_rsids_list = []

    for data in tqdm(testloader):
        inputs1, inputs2, inputs3, inputs4, labels, rsids, query_snp_rsids, metadata = data

        if torch.cuda.is_available():
            inputs1, inputs2, inputs3, inputs4, labels, metadata = Variable(inputs1.cuda()), Variable(inputs2.cuda()), Variable(inputs3.cuda()), Variable(inputs4.cuda()), Variable(labels.cuda()), Variable(metadata.cuda())
        else:
            inputs1, inputs2, inputs3, inputs4, labels, metadata = Variable(inputs1), Variable(inputs2), Variable(inputs3), Variable(inputs4), Variable(labels), Variable(metadata)

        outputs = model(inputs1, inputs2, inputs3, inputs4, metadata)
        _, preds = torch.max(outputs.data, 1)
        probs = F.softmax(outputs, 1)
        correct += torch.sum(preds == labels.data)
        tp += torch.sum((preds + labels.data) == 2)
        size += len(labels)

        probs_list.extend(probs.data[:,1])
        labels_list.extend(labels.data)
        rsids_list.extend(rsids)
        query_snp_rsids_list.extend(query_snp_rsids)

    fpr, tpr, threshold_roc = metrics.roc_curve(labels_list, probs_list)
    precision, recall, threshold_prc = metrics.precision_recall_curve(labels_list, probs_list)
    avg_correct = correct.cpu().numpy() / size
    avg_tp = tp.cpu().numpy() / size
    auroc = metrics.roc_auc_score(labels_list, probs_list)
    auprc = metrics.average_precision_score(labels_list, probs_list)
    print("Average correctness: {:.4f}, Average sensitivity: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}".format(avg_correct, avg_tp, auroc, auprc))

    if sampling == 'locus':
        testset_pred = pd.DataFrame({'rsid': rsids_list, 'query_snp_rsid': query_snp_rsids_list, 'prob': probs_list, 'label': labels_list})
        rank_list = []
        for rs in testset_pred[testset_pred['label']==1]['rsid']:
            rsids_locus = testset_pred[testset_pred['query_snp_rsid'] == rs]
            rsids_locus['rank'] = ss.rankdata(rsids_locus['prob'])
            rank = rsids_locus[rsids_locus['label']==1]['rank'].tolist()
            rank = rank[::-1]
            rank_list.extend(rank)
        avgrank = np.mean(rank_list)
        print('AVGRANK: {}'.format(avgrank))

    # save testing auroc log
    with open(filename_roc,"w") as f:
        f.write('TP\tFP\tthreshold\n')
        for (fp,tp,thr) in zip(fpr,tpr,threshold_roc):
            f.write("{}\t{}\t{}\n".format(fp,tp,thr))
    print('[INFO] testing auroc log saved to {}'.format(filename_roc))

    # save testing auprc log
    with open(filename_prc,"w") as f:
        f.write('Precision\tRecall\tthreshold\n')
        for (pre,rec,thr) in zip(precision,recall,threshold_prc):
            f.write("{}\t{}\t{}\n".format(pre,rec,thr))
    print('[INFO] testing auprc log saved to {}'.format(filename_prc))

    # save testing output
    with open(filename_output, 'w') as f:
        if sampling == 'snp':
            f.write('Acc\tSensitivity\tAUROC\tAUPRC\n')
            f.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(avg_correct, avg_tp, auroc, auprc))
        if sampling == 'locus':
            f.write('Acc\tSensitivity\tAUROC\tAUPRC\tAVGRANK\n')
            f.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(avg_correct, avg_tp, auroc, auprc, avgrank))
        f.write('-'*10 + '\n')
        f.write('test_kwargs: {}\n'.format(test_kwargs))
        f.write('filename_model: {}\n'.format(filename_model))
        f.write('valset: {}\n'.format(filename_testset))
        f.write('sampling: {}\n'.format(sampling))
        f.write('out: {}\n'.format(filename_output))
        f.write('roc: {}\n'.format(filename_roc))
        f.write('prc: {}\n'.format(filename_prc))
        f.write('-'*10+'\n')
        f.write(str(args))

    print('[INFO] testing performance output saved to {}'.format(filename_output))
