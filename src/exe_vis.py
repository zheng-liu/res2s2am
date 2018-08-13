import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter

plt.style.use('ggplot')
sns.set(color_codes=True)
sns.set(font_scale=4)
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set(style='white', palette='muted', color_codes=True)
sns.despine(left=True)

fig_format = 'pdf'
if not os.path.exists('fig'):
    os.system('mkdir fig')

##########
# figure 1
##########
title_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
filename_dataset = 'snap_data_w_annot_1000_seq_reg.txt'
dataset = pd.read_csv(filename_dataset, sep='\t')
d1 = dataset[dataset['label']==1]
d0 = dataset[dataset['label']==0]

# AFR, AMR, ASN, EUR
for i, ft in enumerate(['AFR', 'AMR', 'ASN', 'EUR']):
    f = plt.figure()
    x_range = np.linspace(0.0, 1.0, 11)
    ft_ll1 = np.histogram(d1[ft], x_range, density=True)[0]
    ft_ll0 = np.histogram(d0[ft], x_range, density=True)[0]
    ft_llr = np.log(ft_ll1 / ft_ll0)
    sns.barplot(x=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], y=ft_llr, palette='vlag')
    plt.xlabel('allele frequency', fontsize=18)
    plt.ylabel('LLR(+/-)', fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.title('({}) Binned log-likelihood-ratio: {}'.format(title_list[i], ft), fontsize=18)
    plt.tight_layout()
    plt.savefig('./fig/fig1_{}.{}'.format(ft, fig_format))
    print('[INFO]fig1_{}.{} saved to ./fig'.format(ft, fig_format))

# reg_score_int
f = plt.figure()
ft = 'reg_score_int'
ft_ll1 = np.array([d1[d1[ft] == x].shape[0] for x in np.arange(1.0, 17.0, 1.0)])
ft_ll0 = np.array([d0[d0[ft] == x].shape[0] for x in np.arange(1.0, 17.0, 1.0)])
ft_llr = np.log(ft_ll1 / ft_ll0)
sns.barplot(x=np.arange(1, 17, 1), y=ft_llr, palette='vlag')
plt.xlabel('RegulomeDB score (encoded)', fontsize=18)
plt.ylabel('LLR(+/-)', fontsize=18)
plt.tick_params(axis='both', labelsize=18)
plt.title('(e) Log-likelihood-ratio: {}'.format(ft), fontsize=18)
plt.tight_layout()
plt.savefig('./fig/fig1_{}.{}'.format(ft, fig_format))
print('[INFO]fig1_{}.{} saved to ./fig'.format(ft, fig_format))

# GENCODE_direction, RefSeq_direction
for i, ft in enumerate(['GENCODE_direction', 'RefSeq_direction']):
    f = plt.figure()
    x_range = [-1, 1, 4, 6]
    ft_ll1 = np.histogram(d1[ft], x_range, density=True)[0]
    ft_ll0 = np.histogram(d0[ft], x_range, density=True)[0]
    ft_llr = np.log(ft_ll1 / ft_ll0)
    sns.barplot(x=[0, 3, 5], y=ft_llr, palette='vlag', label=ft)
    plt.xlabel(ft, fontsize=18)
    plt.ylabel('LLR(+/-)', fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.title('({}) Log-likelihood-ratio: {}'.format(title_list[i+5], ft), fontsize=18)
    plt.tight_layout()
    plt.savefig('./fig/fig1_{}.{}'.format(ft, fig_format))
    print('[INFO]fig1_{}.{} saved to ./fig'.format(ft, fig_format))

# GERP_cons, SiPhy_cons
for i, ft in enumerate(['GERP_cons', 'SiPhy_cons']):
    f = plt.figure()
    ft_ll1 = np.array([d1[d1[ft]==0.0].shape[0], d1[d1[ft]==1.0].shape[0]])
    ft_ll0 = np.array([d0[d0[ft]==0.0].shape[0], d0[d0[ft]==1.0].shape[0]])
    ft_llr = np.log(ft_ll1 / ft_ll0)
    sns.barplot(x=[0.0, 1.0], y=ft_llr, palette='vlag')
    plt.xlabel(ft, fontsize=18)
    plt.ylabel('LLR(+/-)', fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.title('({}) Log-likelihood-ratio: {}'.format(title_list[i+7], ft), fontsize=18)
    plt.tight_layout()
    plt.savefig('./fig/fig1_{}.{}'.format(ft, fig_format))
    print('[INFO]fig1_{}.{} saved to ./fig'.format(ft, fig_format))



# plot annotation distribution
f = plt.figure()
table0 = d0.pivot_table(index='chr', columns='annotation', values='label', aggfunc='count')
sns.heatmap(table0, annot=True, cmap='YlGnBu', fmt='g', cbar_kws={'label': 'count'})
f.savefig('./fig/fig1_annot0.{}'.format(fig_format))
print('[INFO] fig1_annot0.{} saved to ./fig'.format(fig_format))

f = plt.figure()
table1 = d1.pivot_table(index='chr', columns='annotation', values='label', aggfunc=np.sum)
sns.heatmap(table1, annot=True, cmap='YlGnBu', fmt='g', cbar_kws={'label': 'count'})
f.savefig('./fig/fig1_annot1.{}'.format(fig_format))
print('[INFO] fig1_annot1.{} saved to ./fig'.format(fig_format))

##########
# figure 2
##########
method_list = ['regulomedb', 'DeFine0', 'DeFine', 'cnn_1s', 'cnn_2s', 'resnet_2s2a', 'resnet_2s2a_metadata']
method_list_xticks = ['RDB', 'DF0', 'DF', 'CNN1s', 'CNN2s', 'Res', 'ResM']
random_seed_list = [1337, 1338, 1339, 1340, 1341]

perf_list = []
for method in method_list:
    for random_seed in random_seed_list:
        filename = 'out_test_{}_1000_fold_{}.txt'.format(method, random_seed)
        perf = pd.read_csv(filename, sep='\t', nrows=1)
        perf['random_seed'] = random_seed
        perf['method'] = method
        perf_list.append(perf)
perf_list = pd.concat(perf_list)

f = plt.figure()
g = sns.boxplot(x='method', y='AUROC', data=perf_list, palette='RdBu')
g.set_xlabel('method', fontsize=15)
g.set_ylabel('AUROC', fontsize=15)
plt.title('(c) 5-fold AUROC', fontsize=15)
g.set_xticklabels(labels=method_list_xticks, rotation=20)
g.tick_params(axis='both', labelsize=15)
g.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.tight_layout()
f.savefig('./fig/fig2_auroc.{}'.format(fig_format))
print('[INFO] ./fig/fig2_auroc.{} saved'.format(fig_format))

f = plt.figure()
g = sns.boxplot(x='method', y='AUPRC', data=perf_list, palette='RdBu')
g.set_xlabel('method', fontsize=15)
g.set_ylabel('AUPRC', fontsize=15)
plt.title('(d) 5-fold AUPRC', fontsize=15)
g.set_xticklabels(labels=method_list_xticks, rotation=20)
g.tick_params(axis='both', labelsize=15)
g.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.tight_layout()
f.savefig('./fig/fig2_auprc.{}'.format(fig_format))
print('[INFO] ./fig/fig2_auprc.{} saved'.format(fig_format))

roc_list = {}
prc_list = {}
tprs = []
precisions = []
recalls = []

for method in method_list:
    roc_list[method] = []
    prc_list[method] = []
    for random_seed in random_seed_list:
        filename_roc = 'roc_test_{}_1000_fold_{}.txt'.format(method, random_seed)
        filename_prc = 'prc_test_{}_1000_fold_{}.txt'.format(method, random_seed)
        roc = pd.read_csv(filename_roc, sep='\t')
        prc = pd.read_csv(filename_prc, sep='\t')
        roc_list[method].append(roc)
        prc_list[method].append(prc)

f = plt.figure()
for i, method in enumerate(method_list):
    for roc in roc_list[method]:
        mean_fp = np.linspace(0, 1, 100)
        tprs.append(interp(mean_fp, roc['FP'], roc['TP']))
        tprs[-1][0] = 0.0

    mean_tp = np.mean(tprs, axis=0)
    mean_tp[-1] = 1.0
    #mean_auc = auc(mean_tp, mean_fp)
    mean_auc = perf_list[perf_list['method']==method]['AUROC'].mean()
    plt.plot(mean_tp, mean_fp, label='{} (Mean AUC={:.4f})'.format(method_list_xticks[i], mean_auc), lw=1, alpha=.8)
    #plt.plot(roc_list[method][0]['TP'], roc_list[method][0]['FP'], label='{} (Mean AUC={:.4f})'.format(method, mean_auc), lw=1, alpha=.8)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
g.set_xticklabels(labels=method_list_xticks)
g.tick_params(axis='both', labelsize=18)
plt.title('(a) ROC curve', fontsize=15)
plt.legend()
f.savefig('./fig/fig2_roc.{}'.format(fig_format))
print('[INFO] fig2_roc.{} saved!'.format(fig_format))

f = plt.figure()
for i, method in enumerate(method_list):
    for prc in prc_list[method]:
        # mean_recall = np.linspace(0, 1, 100)
        mean_threshold = np.linspace(0.0, 0.99, 100)
        precisions.append(interp(mean_threshold, prc['threshold'], prc['Precision']))
        recalls.append(interp(mean_threshold, prc['threshold'], prc['Recall']))
        precisions[-1][0] = 0.0
        recalls[-1][0] = 1.0
        recalls[-1][-1] = 0.0

    mean_precision = np.mean(precisions, axis=0)
    mean_recall = np.mean(recalls, axis=0)
    mean_precision[-1] = 1.0
    mean_auc = perf_list[perf_list['method']==method]['AUPRC'].mean()
    plt.plot(mean_recall, mean_precision, label='{} (Mean AUC={:.4f})'.format(method_list_xticks[i], mean_auc), lw=1, alpha=.8)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)
g.tick_params(axis='both', labelsize=18)
plt.title('(b) PRC curve', fontsize=15)
plt.legend()
f.savefig('./fig/fig2_prc.{}'.format(fig_format))
print('[INFO] fig2_prc.{} saved!'.format(fig_format))

##########
# figure 3
##########
method_list = ['DeFine0', 'DeFine', 'cnn_1s', 'cnn_2s', 'resnet_2s2a', 'resnet_2s2a_metadata']
method_list_xticks = ['DeFine0', 'DeFine', 'CNN_1s', 'CNN_2s', 'Res2s2a', 'Res2s2aM']
random_seed_list = [1337, 1338, 1339, 1340, 1341]

log_list = {}
for method in method_list:
    log_list[method] = []
    for random_seed in random_seed_list:
        filename = 'log_train_{}_1000_fold_{}.txt'.format(method, random_seed)
        log = pd.read_csv(filename, sep='\t')
        log = log[log['phase']=='val']
        log['random_seed'] = random_seed
        log['method'] = method
        log_list[method].append(log)


f = plt.figure()
for i, method in enumerate(method_list):
    loss = np.mean([log['loss'] for log in log_list[method]], axis=0)
    plt.plot(loss, label='{} loss'.format(method_list_xticks[i]), lw=1, alpha=.8)
# plt.xlim([0, 40])
plt.ylim([0.5, 0.9])
plt.xlabel('epoch', fontsize=15)
plt.ylabel('validation loss', fontsize=15)
plt.title('Validation loss vs Epoch', fontsize=15)
g.tick_params(axis='both', labelsize=18)
plt.legend()
f.savefig('./fig/fig2_loss.{}'.format(fig_format))
print('[INFO] fig2_loss.{} saved!'.format(fig_format))
