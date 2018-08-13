# python3 test_regulomedb.py --test_kwargs='test_regulomedb_1000_fold_1337.p' --testset='testset_snp_1000_fold_1337.txt' --sampling='snp' --output='out_test_regulomedb_1000_fold_1337.txt' --roc='roc_test_regulomedb_1000_fold_1337.txt' --prc='prc_test_regulomedb_1000_fold_1337.txt'
# python3 test_regulomedb.py --test_kwargs='test_regulomedb_1000_fold_1338.p' --testset='testset_snp_1000_fold_1338.txt' --sampling='snp' --output='out_test_regulomedb_1000_fold_1338.txt' --roc='roc_test_regulomedb_1000_fold_1338.txt' --prc='prc_test_regulomedb_1000_fold_1338.txt'
# python3 test_regulomedb.py --test_kwargs='test_regulomedb_1000_fold_1339.p' --testset='testset_snp_1000_fold_1339.txt' --sampling='snp' --output='out_test_regulomedb_1000_fold_1339.txt' --roc='roc_test_regulomedb_1000_fold_1339.txt' --prc='prc_test_regulomedb_1000_fold_1339.txt'
# python3 test_regulomedb.py --test_kwargs='test_regulomedb_1000_fold_1340.p' --testset='testset_snp_1000_fold_1340.txt' --sampling='snp' --output='out_test_regulomedb_1000_fold_1340.txt' --roc='roc_test_regulomedb_1000_fold_1340.txt' --prc='prc_test_regulomedb_1000_fold_1340.txt'
# python3 test_regulomedb.py --test_kwargs='test_regulomedb_1000_fold_1341.p' --testset='testset_snp_1000_fold_1341.txt' --sampling='snp' --output='out_test_regulomedb_1000_fold_1341.txt' --roc='roc_test_regulomedb_1000_fold_1341.txt' --prc='prc_test_regulomedb_1000_fold_1341.txt'

# python3 test_regulomedb.py --test_kwargs='test_regulomedb_locus_1000_fold_1337.p' --testset='testset_locus_1000_fold_1337.txt' --sampling='locus' --output='out_test_regulomedb_locus_1000_fold_1337.txt' --roc='roc_test_regulomedb_locus_1000_fold_1337.txt' --prc='prc_test_regulomedb_locus_1000_fold_1337.txt'
# python3 test_regulomedb.py --test_kwargs='test_regulomedb_locus_1000_fold_1338.p' --testset='testset_locus_1000_fold_1338.txt' --sampling='locus' --output='out_test_regulomedb_locus_1000_fold_1338.txt' --roc='roc_test_regulomedb_locus_1000_fold_1338.txt' --prc='prc_test_regulomedb_locus_1000_fold_1338.txt'
# python3 test_regulomedb.py --test_kwargs='test_regulomedb_locus_1000_fold_1339.p' --testset='testset_locus_1000_fold_1339.txt' --sampling='locus' --output='out_test_regulomedb_locus_1000_fold_1339.txt' --roc='roc_test_regulomedb_locus_1000_fold_1339.txt' --prc='prc_test_regulomedb_locus_1000_fold_1339.txt'
# python3 test_regulomedb.py --test_kwargs='test_regulomedb_locus_1000_fold_1340.p' --testset='testset_locus_1000_fold_1340.txt' --sampling='locus' --output='out_test_regulomedb_locus_1000_fold_1340.txt' --roc='roc_test_regulomedb_locus_1000_fold_1340.txt' --prc='prc_test_regulomedb_locus_1000_fold_1340.txt'
# python3 test_regulomedb.py --test_kwargs='test_regulomedb_locus_1000_fold_1341.p' --testset='testset_locus_1000_fold_1341.txt' --sampling='locus' --output='out_test_regulomedb_locus_1000_fold_1341.txt' --roc='roc_test_regulomedb_locus_1000_fold_1341.txt' --prc='prc_test_regulomedb_locus_1000_fold_1341.txt'

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

def make_kwargs(filename_test_kwargs):
    test_kwargs_reg_score = {}
    test_kwargs_reg_score['random_seed'] = 1337
    test_kwargs_reg_score['shuffle'] = True
    pickle.dump(test_kwargs_reg_score, open(filename_test_kwargs, 'wb'))
    print('[INFO] {} dumped!'.format(filename_test_kwargs))
    return test_kwargs_reg_score

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--test_kwargs', help='kwargs of test model.')
    argparser.add_argument('--testset', help='testset file name.')
    argparser.add_argument('--sampling', choices=['snp', 'locus'], help='Train, validation, testing sampling method.', required=True)
    argparser.add_argument('--model', help='model file name.')
    argparser.add_argument('--output', help='output file name.')
    argparser.add_argument('--roc', help='roc file name')
    argparser.add_argument('--prc', help='prc file name')
    args = argparser.parse_args()

    filename_test_kwargs = args.test_kwargs
    test_kwargs = make_kwargs(filename_test_kwargs)
    filename_testset = args.testset
    filename_model = args.model
    filename_output = args.output
    filename_roc = args.roc
    filename_prc = args.prc
    sampling = args.sampling

    random_seed = test_kwargs['random_seed']
    shuffle = test_kwargs['shuffle']

    testset = pd.read_csv(filename_testset, sep='\t')
    prob_label = testset[['reg_score_int', 'label']]
    print(prob_label)
    print(prob_label.shape)
    # prob_label = prob_label[prob_label['reg_score_int'] != 0] # exclude the NaN reg_score
    prob_label['prob'] = prob_label['reg_score_int'].apply(lambda x: 1.0 / 16.0 * (17.0 - x))
    fpr, tpr, threshold_roc = metrics.roc_curve(prob_label['label'], prob_label['prob'])
    precision, recall, threshold_prc = metrics.precision_recall_curve(prob_label['label'], prob_label['prob'])
    auroc = metrics.roc_auc_score(prob_label['label'], prob_label['prob'])
    auprc = metrics.average_precision_score(prob_label['label'], prob_label['prob'])
    print("AUROC: {:.4f}, AUPRC: {:.4f}".format(auroc, auprc))

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
            f.write('AUROC\tAUPRC\n')
            f.write('{:.4f}\t{:.4f}\n'.format(auroc, auprc))
        if sampling == 'locus':
            f.write('AUROC\tAUPRC\tAVGRANK\n')
            f.write('{:.4f}\t{:.4f}\t{:.4f}\n'.format(auroc, auprc, avgrank))
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
