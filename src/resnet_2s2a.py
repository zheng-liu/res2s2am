import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def conv1x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class resnet_2s2a(nn.Module):
    def __init__(self, kwargs):
        super(resnet_2s2a, self).__init__()
        block = kwargs['block']
        layers = kwargs['layers']
        seqlength = kwargs['seqlength']
        num_classes = kwargs['num_classes']
        self.inplanes_ref = 16
        self.inplanes_alt = 16
        self.conv1_ref = nn.Conv1d(4, 16, kernel_size=4, stride=2, padding=3, bias=False)
        self.conv1_alt = nn.Conv1d(4, 16, kernel_size=4, stride=2, padding=3, bias=False)
        self.bn1_ref = nn.BatchNorm1d(16)
        self.bn1_alt = nn.BatchNorm1d(16)
        self.relu_ref = nn.ReLU(inplace=True)
        self.relu_alt = nn.ReLU(inplace=True)
        self.maxpool_ref = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.maxpool_alt = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1_ref = self._make_layer_ref(block, 16, layers[0])
        self.layer1_alt = self._make_layer_alt(block, 16, layers[0])
        self.layer2_ref = self._make_layer_ref(block, 32, layers[1], stride=2)
        self.layer2_alt = self._make_layer_alt(block, 32, layers[1], stride=2)
        self.layer3_ref = self._make_layer_ref(block, 64, layers[2], stride=2)
        self.layer3_alt = self._make_layer_alt(block, 64, layers[2], stride=2)
        self.layer4_ref = self._make_layer_ref(block, 128, layers[3], stride=2)
        self.layer4_alt = self._make_layer_alt(block, 128, layers[3], stride=2)
        self.avgpool_ref = nn.AvgPool1d(4, stride=1)
        self.avgpool_alt = nn.AvgPool1d(4, stride=1)
        self.seqlength = seqlength
        self.outlength_ref = math.ceil(math.ceil((math.ceil(math.ceil(math.ceil(self.seqlength / self.conv1_ref.stride[0]) / self.maxpool_ref.stride) / self.layer1_ref[0].stride) / self.layer2_ref[0].stride) / self.layer3_ref[0].stride) / self.layer4_ref[0].stride) - self.avgpool_ref.kernel_size[0] + 1
        self.fc = nn.Linear(128 * block.expansion * self.outlength_ref * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_ref(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_ref != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes_ref, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_ref, planes, stride, downsample))
        self.inplanes_ref = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_ref, planes))

        return nn.Sequential(*layers)

    def _make_layer_alt(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_alt != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes_alt, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_alt, planes, stride, downsample))
        self.inplanes_alt = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_alt, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3, x4):
        x1 = self.conv1_ref(x1)
        x2 = self.conv1_ref(x2)
        x3 = self.conv1_alt(x3)
        x4 = self.conv1_alt(x4)
        x1 = self.bn1_ref(x1)
        x2 = self.bn1_ref(x2)
        x3 = self.bn1_alt(x3)
        x4 = self.bn1_alt(x4)
        x1 = self.relu_ref(x1)
        x2 = self.relu_ref(x2)
        x3 = self.relu_alt(x3)
        x4 = self.relu_alt(x4)
        x1 = self.maxpool_ref(x1)
        x2 = self.maxpool_ref(x2)
        x3 = self.maxpool_alt(x3)
        x4 = self.maxpool_alt(x4)
        x1 = self.layer1_ref(x1)
        x2 = self.layer1_ref(x2)
        x3 = self.layer1_alt(x3)
        x4 = self.layer1_alt(x4)
        x1 = self.layer2_ref(x1)
        x2 = self.layer2_ref(x2)
        x3 = self.layer2_alt(x3)
        x4 = self.layer2_alt(x4)
        x1 = self.layer3_ref(x1)
        x2 = self.layer3_ref(x2)
        x3 = self.layer3_alt(x3)
        x4 = self.layer3_alt(x4)
        x1 = self.layer4_ref(x1)
        x2 = self.layer4_ref(x2)
        x3 = self.layer4_alt(x3)
        x4 = self.layer4_alt(x4)
        x1 = self.avgpool_ref(x1)
        x2 = self.avgpool_ref(x2)
        x3 = self.avgpool_alt(x3)
        x4 = self.avgpool_alt(x4)

        x1 = x1.view(-1, self.num_flat_features(x1))
        x2 = x2.view(-1, self.num_flat_features(x2))
        x3 = x3.view(-1, self.num_flat_features(x3))
        x4 = x4.view(-1, self.num_flat_features(x4))

        x_ref = torch.cat([x1, x2], 1)
        x_alt = torch.cat([x3, x4], 1)
        x = x_ref - x_alt

        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":
    torch.manual_seed(1337)
    kwargs = {'block': Bottleneck, 'layers':[3,4,23,3], 'seqlength': 1000, 'num_classes': 2}
    net = resnet_2s2a(kwargs)
    input1 = Variable(torch.randn(1, 4, 1000))
    input2 = Variable(torch.randn(1, 4, 1000))
    input3 = Variable(torch.randn(1, 4, 1000))
    input4 = Variable(torch.randn(1, 4, 1000))
    output = net(input1, input2, input3, input4)
    print(output)
