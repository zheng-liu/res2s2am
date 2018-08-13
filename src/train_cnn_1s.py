# python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_1000_fold_1337.p' --train_kwargs='train_kwargs_cnn_1s_1000_fold_1337.p' --trainset='trainset_snp_1000_fold_1337.txt' --valset='valset_snp_1000_fold_1337.txt' --sampling='snp' --output='out_train_cnn_1s_1000_fold_1337.txt' --log='log_train_cnn_1s_1000_fold_1337.txt' --model='model_cnn_1s_1000_fold_1337.pt'
# python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_1000_fold_1338.p' --train_kwargs='train_kwargs_cnn_1s_1000_fold_1338.p' --trainset='trainset_snp_1000_fold_1338.txt' --valset='valset_snp_1000_fold_1338.txt' --sampling='snp' --output='out_train_cnn_1s_1000_fold_1338.txt' --log='log_train_cnn_1s_1000_fold_1338.txt' --model='model_cnn_1s_1000_fold_1338.pt'
# python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_1000_fold_1339.p' --train_kwargs='train_kwargs_cnn_1s_1000_fold_1339.p' --trainset='trainset_snp_1000_fold_1339.txt' --valset='valset_snp_1000_fold_1339.txt' --sampling='snp' --output='out_train_cnn_1s_1000_fold_1339.txt' --log='log_train_cnn_1s_1000_fold_1339.txt' --model='model_cnn_1s_1000_fold_1339.pt'
# python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_1000_fold_1340.p' --train_kwargs='train_kwargs_cnn_1s_1000_fold_1340.p' --trainset='trainset_snp_1000_fold_1340.txt' --valset='valset_snp_1000_fold_1340.txt' --sampling='snp' --output='out_train_cnn_1s_1000_fold_1340.txt' --log='log_train_cnn_1s_1000_fold_1340.txt' --model='model_cnn_1s_1000_fold_1340.pt'
# python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_1000_fold_1341.p' --train_kwargs='train_kwargs_cnn_1s_1000_fold_1341.p' --trainset='trainset_snp_1000_fold_1341.txt' --valset='valset_snp_1000_fold_1341.txt' --sampling='snp' --output='out_train_cnn_1s_1000_fold_1341.txt' --log='log_train_cnn_1s_1000_fold_1341.txt' --model='model_cnn_1s_1000_fold_1341.pt'

# python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_locus_1000_fold_1337.p' --train_kwargs='train_kwargs_cnn_1s_locus_1000_fold_1337.p' --trainset='trainset_locus_1000_fold_1337.txt' --valset='valset_locus_1000_fold_1337.txt' --sampling='locus' --output='out_train_cnn_1s_locus_1000_fold_1337.txt' --log='log_train_cnn_1s_locus_1000_fold_1337.txt' --model='model_cnn_1s_locus_1000_fold_1337.pt'
# python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_locus_1000_fold_1338.p' --train_kwargs='train_kwargs_cnn_1s_locus_1000_fold_1338.p' --trainset='trainset_locus_1000_fold_1338.txt' --valset='valset_locus_1000_fold_1338.txt' --sampling='locus' --output='out_train_cnn_1s_locus_1000_fold_1338.txt' --log='log_train_cnn_1s_locus_1000_fold_1338.txt' --model='model_cnn_1s_locus_1000_fold_1338.pt'
# python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_locus_1000_fold_1339.p' --train_kwargs='train_kwargs_cnn_1s_locus_1000_fold_1339.p' --trainset='trainset_locus_1000_fold_1339.txt' --valset='valset_locus_1000_fold_1339.txt' --sampling='locus' --output='out_train_cnn_1s_locus_1000_fold_1339.txt' --log='log_train_cnn_1s_locus_1000_fold_1339.txt' --model='model_cnn_1s_locus_1000_fold_1339.pt'
# python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_locus_1000_fold_1340.p' --train_kwargs='train_kwargs_cnn_1s_locus_1000_fold_1340.p' --trainset='trainset_locus_1000_fold_1340.txt' --valset='valset_locus_1000_fold_1340.txt' --sampling='locus' --output='out_train_cnn_1s_locus_1000_fold_1340.txt' --log='log_train_cnn_1s_locus_1000_fold_1340.txt' --model='model_cnn_1s_locus_1000_fold_1340.pt'
# python3 train_cnn_1s.py --device=? --model_kwargs='model_kwargs_cnn_1s_locus_1000_fold_1341.p' --train_kwargs='train_kwargs_cnn_1s_locus_1000_fold_1341.p' --trainset='trainset_locus_1000_fold_1341.txt' --valset='valset_locus_1000_fold_1341.txt' --sampling='locus' --output='out_train_cnn_1s_locus_1000_fold_1341.txt' --log='log_train_cnn_1s_locus_1000_fold_1341.txt' --model='model_cnn_1s_locus_1000_fold_1341.pt'


import cnn_1s
import copy
import pickle
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from sklearn import metrics
from dataloader import GenomeDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

def make_kwargs(filename_model_kwargs, filename_train_kwargs):
    # model_kwargs, cnn_1s
    model_kwargs_cnn_1s = {}
    seqlen = 1000
    conv1 = {'in_channels': 4, 'out_channels': 16, 'kernel_size': 3,
             'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True}
    conv2 = {'in_channels': 16, 'out_channels': 32, 'kernel_size': 5,
             'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True}
    conv3 = {'in_channels': 32, 'out_channels': 64, 'kernel_size': 7,
             'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1, 'bias': True}
    maxpool1 = {'kernel_size': 4, 'stride': 4}
    maxpool2 = {'kernel_size': 4, 'stride': 4}
    dropout1 = {'p': 0.2}
    dropout2 = {'p': 0.3}
    dropout3 = {'p': 0.5}
    fc1 = {'out_channels': 100}
    fc2 = {'out_channels': 2}
    model_kwargs_cnn_1s = {'conv1': conv1, 'conv2': conv2, 'conv3': conv3, \
                       'maxpool1': maxpool1, 'maxpool2': maxpool2, \
                       'dropout1': dropout1, 'dropout2': dropout2, \
                       'dropout3': dropout3, 'fc1': fc1, 'fc2': fc2, \
                       'seqlen': seqlen}
    pickle.dump(model_kwargs_cnn_1s, open(filename_model_kwargs, 'wb'))
    print('[INFO] {} dumped!'.format(filename_model_kwargs))

    # train_kwargs, cnn_1s
    train_kwargs_cnn_1s = {}
    train_kwargs_cnn_1s['random_seed'] = 1337
    # train_kwargs_cnn_1s['device'] = 0 # [0,1,2,3,4,5,6,7]
    train_kwargs_cnn_1s['optim'] = 'Adam' # ['SGD', 'Adam', 'Adagrad', 'RMSProp']
    train_kwargs_cnn_1s['optim_param'] = {'betas': [0.9, 0.999], 'lr': 0.001, 'weight_decay': 1e-5}
    train_kwargs_cnn_1s['scheduler'] = 'StepLR' # ['StepLR', 'MultiStepLR', 'ReduceLROnPlateau']
    train_kwargs_cnn_1s['scheduler_param'] = {'step_size': 10, 'gamma': 0.1}
    train_kwargs_cnn_1s['loss'] = 'CrossEntropyLoss' # ['CrossEntropyLoss', 'NLLLoss', 'MSELoss']
    train_kwargs_cnn_1s['cudnn'] = True
    train_kwargs_cnn_1s['imbalance'] = [1.0, 10.0]
    train_kwargs_cnn_1s['batch_size'] = 256
    train_kwargs_cnn_1s['encode_mode'] = 'N_2_zero'
    # train_kwargs_cnn_1s['num_epochs'] = 40
    train_kwargs_cnn_1s['num_epochs'] = 20
    train_kwargs_cnn_1s['shuffle'] = True
    train_kwargs_cnn_1s['num_workers'] = 8
    pickle.dump(train_kwargs_cnn_1s, open(filename_train_kwargs, 'wb'))
    print('[INFO] {} dumped!'.format(filename_train_kwargs))

    return model_kwargs_cnn_1s, train_kwargs_cnn_1s

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_kwargs', help='kwargs of network model.')
    argparser.add_argument('--train_kwargs', help='kwargs of network model training.')
    argparser.add_argument('--trainset', help='trainset file name.')
    argparser.add_argument('--valset', help='valset file name.')
    argparser.add_argument('--sampling', choices=['snp', 'locus'], help='Train, validation, testing sampling method.', required=True)
    argparser.add_argument('--model', help='model file name.')
    argparser.add_argument('--output', help='output file name.')
    argparser.add_argument('--log', help='log file name.')
    argparser.add_argument('--device', help='device number.')
    args = argparser.parse_args()

    filename_model_kwargs = args.model_kwargs
    filename_train_kwargs = args.train_kwargs
    model_kwargs, train_kwargs = make_kwargs(filename_model_kwargs, filename_train_kwargs)
    sampling = args.sampling
    filename_trainset = args.trainset
    filename_valset = args.valset
    filename_model = args.model
    filename_output = args.output
    filename_log = args.log
    device = int(args.device)

    random_seed = train_kwargs['random_seed']
    #device = train_kwargs['device'] # [0,1,2,3,4,5,6,7]
    optim = train_kwargs['optim'] # ['SGD', 'Adam', 'Adagrad', 'RMSProp']
    optim_param = train_kwargs['optim_param'] # {'betas': [0.9, 0.999], 'lr': 0.001, 'weight_decay': 1e-5}
    scheduler =  train_kwargs['scheduler'] # ['StepLR', 'MultiStepLR', 'ReduceLROnPlateau']
    scheduler_param = train_kwargs['scheduler_param'] # {'step_size': 10, 'gamma': 0.1}
    loss = train_kwargs['loss'] # ['CrossEntropyLoss', 'NLLLoss', 'MSELoss']
    cudnn = train_kwargs['cudnn'] # [True, False]
    imbalance = train_kwargs['imbalance'] # 10.0
    batch_size = train_kwargs['batch_size'] # 16, .. 32
    encode_mode = train_kwargs['encode_mode'] # 'N_2_zero', 'N_2_quarter'
    num_epochs = train_kwargs['num_epochs'] # 10
    shuffle = train_kwargs['shuffle']
    num_workers = train_kwargs['num_workers']

    torch.cuda.set_device(device) # assign gpu
    gpu = torch.cuda.is_available()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.fastest = cudnn
    torch.manual_seed(random_seed)

    if gpu:
        model = cnn_1s.cnn_1s(model_kwargs).cuda()
        #model = torch.nn.DataParallel(model)
        class_imbalance = torch.FloatTensor(imbalance).cuda()
    else:
        model = cnn_1s.cnn_1s(model_kwargs)
        class_imbalance = torch.FloatTensor(imbalance)

    # build optimizer
    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **optim_param)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **optim_param)
    elif optim == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), **optim_param)
    elif optim == 'RMSProp':
        optimizer = torch.optim.RMSProp(model.parameters(), **optim_param)

    # build scheduler
    if scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_param)
    elif scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_param)
    elif scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_param)

    if loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=class_imbalance)
    elif loss == 'NLLLoss':
        criterion = nn.NLLLoss(weight=class_imbalance)
    elif loss == 'MSELoss':
        criterion = nn.MSELoss(weight=class_imbalance)

    # read datasets
    trainset = GenomeDataset(filename=filename_trainset, seq_name=['seq_ref_1'], encode_mode=encode_mode)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    valset = GenomeDataset(filename=filename_valset, seq_name=['seq_ref_1'], encode_mode=encode_mode)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    dataloaders = {'train': trainloader, 'val': valloader}
    print("[INFO] dataloaders generated")
    print('[INFO] start to train and tune')
    print('-' * 10)

    start_time = time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_sensitivity = 0.0
    best_auroc = 0.0
    best_auprc = 0.0
    best_epoch = -1
    best_labels_list = []
    best_probs_list = []
    epoch_loss_min = float('inf')
    epoch_loss_prev = float('inf')
    uptrend = 0 # num of continuous saturated epoches
    max_uptrend = 10 # max num of continuous aturated epoches
    early_stopping = False
    performance_hist = pd.DataFrame(columns=["loss", "acc", "sensitivity", "auroc", "auprc", "epoch", "phase"])

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        dataset_sizes = {'train': 0.0, 'val':0.0} # set to float for later division

        for phase in ['train', 'val']:

            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0.0
            running_tp = 0.0 # num of true positive cases
            labels_list = []
            probs_list = []

            for data in tqdm(dataloaders[phase]):
                inputs, labels, _, _ = data
                if gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                probs = outputs.data[:, 1]
                loss = criterion(outputs, labels)

                labels_list.extend(labels.data)
                probs_list.extend(probs)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_tp += torch.sum((preds + labels.data) == 2)
                dataset_sizes[phase] += len(labels)

            epoch_loss = running_loss.cpu().numpy() / dataset_sizes[phase]
            epoch_acc = running_corrects.cpu().numpy() / dataset_sizes[phase]
            epoch_sensitivity = running_tp.cpu().numpy() / dataset_sizes[phase]

            performance = {'loss': epoch_loss, 'acc': epoch_acc, 'sensitivity': epoch_sensitivity, 'epoch': epoch, 'phase': phase}
            performance_hist = performance_hist.append(performance, ignore_index=True)
            print('{} Loss: {:.4f} ACC: {:.4f}, Sensitivity:{:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_sensitivity))

            if phase == 'val':
                if epoch_loss < epoch_loss_min:
                    best_epoch = epoch
                    epoch_loss_min = epoch_loss
                    best_labels_list = labels_list
                    best_probs_list = probs_list
                    best_acc = epoch_acc
                    best_sensitivity = epoch_sensitivity
                    torch.save(model, filename_model)
                    print('[INFO] save model after {} epoch to {}'.format(epoch, filename_model))

                if epoch_loss < epoch_loss_prev:
                    uptrend = 0
                    epoch_loss_prev = epoch_loss
                else:
                    uptrend += 1
                    epoch_loss_prev = epoch_loss

                if uptrend == max_uptrend:
                    early_stopping = True
                    print('[INFO] loss: {}, acc: {}, sensitivity: {}, AUROC: {}, AUPRC: {}, best_epoch: {}, total_epoch: {}, phase: {}'.format(epoch_loss, best_acc, best_sensitivity, best_auroc, best_auprc, best_epoch, epoch, phase))

                if early_stopping:
                    print('[INFO] early stop')
                    break

        if early_stopping:
            break

    best_auroc = metrics.roc_auc_score(best_labels_list, best_probs_list)
    best_auprc = metrics.average_precision_score(best_labels_list, best_probs_list)

    time_elapsed = time() -start_time
    print('Training completes in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    performance_hist.to_csv(filename_log, sep='\t', index=False)
    print('[INFO] performance_hist saved to {}'.format(filename_log))
    print('[INFO] fine-tuned model saved to {}'.format(filename_model))

    with open(filename_output, 'w') as f:
        f.write('Acc\tSensitivity\tAUROC\tAUPRC\n')
        f.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(best_acc, best_sensitivity, best_auroc, best_auprc))
        f.write('-'*10+'\n')
        f.write('model_kwargs: {}\n'.format(model_kwargs))
        f.write('train_kwargs: {}\n'.format(train_kwargs))
        f.write('trainset: {}\n'.format(filename_trainset))
        f.write('valset: {}\n'.format(filename_valset))
        f.write('model: {}\n'.format(filename_model))
        f.write('sampling: {}\n'.format(sampling))
        f.write('out: {}\n'.format(filename_output))
        f.write('log: {}\n'.format(filename_log))
        f.write('-'*10+'\n')
        f.write(str(args))
    print('[INFO] cnn_1s model training output (with settings) saved to {}'.format(filename_output))
