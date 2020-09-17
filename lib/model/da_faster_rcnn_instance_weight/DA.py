from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.da_faster_rcnn_instance_weight.LabelResizeLayer import (
    ImageLabelResizeLayer,
    InstanceLabelResizeLayer,
)
from model.utils.config import cfg
from torch.autograd import Function, Variable


class GRLayer(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.alpha = 0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output


def grad_reverse(x):
    return GRLayer.apply(x)


class _ImageDA(nn.Module):
    def __init__(self, dim, num_classes):
        super(_ImageDA, self).__init__()
        self.dim = dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=True)
        self.Conv2 = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=True)
        self.reLu = nn.ReLU(inplace=False)
        self.LabelResizeLayer = ImageLabelResizeLayer()

        #for local sub-domain discriminators
        self.conv_lst = nn.Conv2d(self.dim, num_classes-1, kernel_size=1, stride=1, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.num_classes = num_classes-1
        self.dci = {}

        for i in range(num_classes-1):
            self.dci[i] = {}
            self.dci[i][0] = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=True)
            self.dci[i][0].cuda()
            self.dci[i][1] = nn.ReLU(inplace=False)
            self.dci[i][2] = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=True)
            self.dci[i][2].cuda()


    def forward(self, x, need_backprop):
        reverse_x = grad_reverse(x)
        global_x = self.reLu(self.Conv1(reverse_x))
        global_x = self.Conv2(global_x)
        label = self.LabelResizeLayer(global_x, need_backprop)

        #for local sub-domain discriminator
        #p*feature -> classifier_i -> loss_i
        local_out = []
        local_x = self.conv_lst(x)
        local_x = self.softmax(local_x)
        for i in range(self.num_classes):
            ps = local_x[:, i].unsqueeze(1)
            local_reverse_x = ps * reverse_x

            local_out_i = self.dci[i][1](self.dci[i][0](local_reverse_x))
            local_out_i = self.dci[i][2](local_out_i)
            local_out.append(local_out_i)


        return global_x, local_out, label


class _InstanceDA(nn.Module):
    def __init__(self, in_channel=4096):
        super(_InstanceDA, self).__init__()
        self.dc_ip1 = nn.Linear(in_channel, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer = nn.Linear(1024, 1)
        self.LabelResizeLayer = InstanceLabelResizeLayer()

    def forward(self, x, need_backprop):
        x = grad_reverse(x)
        x = self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x = self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x = F.sigmoid(self.clssifer(x))
        label = self.LabelResizeLayer(x, need_backprop)
        return x, label
