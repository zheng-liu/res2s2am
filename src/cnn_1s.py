import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class cnn_1s(nn.Module):
    """ Hyperparameters:
        - conv out_channels (filter number)
        - conv kernel_size (kernel size)
        - conv stride
        - conv padding
        - conv dilation
        - conv groups (separate the conv layers towards output)
    """
    def __init__(self, kwargs):
        super(cnn_1s, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=kwargs["conv1"]["in_channels"], out_channels=kwargs["conv1"]["out_channels"], \
            kernel_size=kwargs["conv1"]["kernel_size"], stride=kwargs["conv1"]["stride"], \
            padding=kwargs["conv1"]["padding"], dilation=kwargs["conv1"]["dilation"], \
            groups=kwargs["conv1"]["groups"], bias=kwargs["conv1"]["bias"])
        self.conv2 = nn.Conv1d(in_channels=kwargs["conv2"]["in_channels"], out_channels=kwargs["conv2"]["out_channels"] , \
            kernel_size=kwargs["conv2"]["kernel_size"], stride=kwargs["conv2"]["stride"], \
            padding=kwargs["conv2"]["padding"], dilation=kwargs["conv2"]["dilation"], \
            groups=kwargs["conv2"]["groups"], bias=kwargs["conv2"]["bias"])
        self.conv3 = nn.Conv1d(in_channels=kwargs["conv3"]["in_channels"], out_channels=kwargs["conv3"]["out_channels"], \
            kernel_size=kwargs["conv3"]["kernel_size"], stride=kwargs["conv3"]["stride"], \
            padding=kwargs["conv3"]["padding"], dilation=kwargs["conv3"]["dilation"], groups=kwargs["conv3"]["groups"], \
            bias=kwargs["conv3"]["bias"])
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool1 = nn.MaxPool1d(kernel_size=kwargs["maxpool1"]["kernel_size"], stride=kwargs["maxpool1"]["stride"])
        self.maxpool2 = nn.MaxPool1d(kernel_size=kwargs["maxpool2"]["kernel_size"], stride=kwargs["maxpool2"]["stride"])
        self.dropout1 = nn.Dropout(p=kwargs["dropout1"]["p"], inplace=False)
        self.dropout2 = nn.Dropout(p=kwargs["dropout2"]["p"], inplace=False)
        self.dropout3 = nn.Dropout(p=kwargs["dropout3"]["p"], inplace=False)
        self.bn1 = nn.BatchNorm1d(kwargs['conv1']['out_channels'])
        self.bn2 = nn.BatchNorm1d(kwargs['conv2']['out_channels'])
        self.bn3 = nn.BatchNorm1d(kwargs['conv3']['out_channels'])
        self.seqlength = kwargs['seqlen']
        self.fc1_in_channels = (math.floor( \
                               (math.floor((self.seqlength - kwargs['conv1']['kernel_size'] + 1) / (kwargs['maxpool1']['kernel_size'] * 1.0)) - \
                               kwargs['conv2']['kernel_size'] + 1 ) / (kwargs['maxpool2']['kernel_size'] * 1.0)) - \
                               kwargs['conv3']['kernel_size'] + 1) * kwargs['conv3']['out_channels']
        self.fc1 = nn.Linear(self.fc1_in_channels, kwargs["fc1"]["out_channels"])
        self.fc2 = nn.Linear(kwargs["fc1"]["out_channels"], kwargs["fc2"]["out_channels"])
        #self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax()

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                # m.weight.data.normal_(0, kwargs['stdv'])
                
            # if isinstance(m, nn.BatchNorm1d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
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
