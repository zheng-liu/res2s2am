import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class DeFine0(nn.Module):
    """ Hyperparameters:
        - conv out_channels (filter number)
        - conv kernel_size (kernel size)
        - conv stride
        - conv padding
        - conv dilation
        - conv groups (separate the conv layers towards output)
    """
    def __init__(self, kwargs):
        super(DeFine0, self).__init__()
        self.seqlength = kwargs['seqlen']
        self.conv1 = nn.Conv1d(in_channels=kwargs["conv1"]["in_channels"], out_channels=kwargs["conv1"]["out_channels"], \
            kernel_size=kwargs["conv1"]["kernel_size"], stride=kwargs["conv1"]["stride"], \
            padding=kwargs["conv1"]["padding"], dilation=kwargs["conv1"]["dilation"], \
            groups=kwargs["conv1"]["groups"], bias=kwargs["conv1"]["bias"])
        self.relu_major = nn.ReLU(inplace=False)
        self.relu_minor = nn.ReLU(inplace=False)
        self.maxpool_major = nn.MaxPool1d(kernel_size=kwargs["maxpool_major"]["kernel_size"], stride=kwargs["maxpool_major"]["stride"])
        self.maxpool_minor = nn.MaxPool1d(kernel_size=kwargs["maxpool_minor"]["kernel_size"], stride=kwargs["maxpool_minor"]["stride"])
        self.avgpool_major = nn.AvgPool1d(kernel_size=kwargs["avgpool_major"]["kernel_size"], stride=kwargs["avgpool_major"]["stride"])
        self.avgpool_minor = nn.AvgPool1d(kernel_size=kwargs["avgpool_minor"]["kernel_size"], stride=kwargs["avgpool_minor"]["stride"])
        self.dropout = nn.Dropout(p=kwargs["dropout"]["p"], inplace=False)
        self.bn_in_channels = math.floor((self.seqlength - kwargs['conv1']['kernel_size'] + 1) / (kwargs['maxpool_major']['kernel_size'] * 1.0)) * kwargs['conv1']['out_channels'] * 4
        self.bn = nn.BatchNorm1d(self.bn_in_channels)
        self.fc1 = nn.Linear(self.bn_in_channels, kwargs["fc1"]["out_channels"])
        self.fc2 = nn.Linear(kwargs["fc1"]["out_channels"], kwargs["fc2"]["out_channels"])
        self.fc3 = nn.Linear(kwargs['fc2']['out_channels'], kwargs['fc3']['out_channels'])

        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.weight.data.normal_(0.0, 1.0)

            # if isinstance(m, nn.BatchNorm1d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def forward(self, x1, x2):

        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        x1 = self.relu_major(x1)
        x2 = self.relu_minor(x2)
        x1_mp = self.maxpool_major(x1)
        x1_ap = self.avgpool_major(x1)
        x2_mp = self.maxpool_major(x2)
        x2_ap = self.avgpool_minor(x2)

        x1_mp = x1_mp.view(-1, self.num_flat_features(x1_mp))
        x1_ap = x1_ap.view(-1, self.num_flat_features(x1_ap))
        x2_mp = x1_mp.view(-1, self.num_flat_features(x2_mp))
        x2_ap = x2_ap.view(-1, self.num_flat_features(x2_ap))

        x = torch.cat([x1_mp, x1_ap, x2_mp, x2_ap], 1)

        x = self.bn(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)

        # Attention: there is no need to add softmax layer since it is included in CrossEntropyLoss!
        # x = self.sigmoid(x)
        #x = self.softmax(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
