from torch import nn
from .models import Bottleneck
from .models import Hourglass
from .models import MessagePassing


class Estimator(nn.Module):

    def __init__(self, stacks=4, msg_pass=1):
        super(Estimator, self).__init__()
        self.stacks = stacks
        self.msg_pass = msg_pass
        self.conv1 = nn.Conv2d(1, 64, padding=3, kernel_size=7,
                               stride=2, bias=False)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv1_relu = nn.ReLU(inplace=False)
        self.pre_res_1 = Bottleneck(64, 32)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)  # problem, need to see the source code of caffe
        self.pre_res_2 = Bottleneck(128, 32)
        self.pre_res_2_bn = nn.BatchNorm2d(128)
        self.pre_res_2_relu = nn.ReLU(inplace=False)
        self.hourglass_0 = Bottleneck(128, 64)
        hg, mp = [], []
        linear_1_res, linear_1_bn, linear_1_relu, linear_1_conv = [], [], [], []
        linear_2_bn, linear_2_relu, linear_2_conv = [], [], []
        linear_3 = []
        linear_mp_bn, linear_mp_relu, linear_mp_conv = [], [], []
        for index in range(self.stacks):
            hg.append(Hourglass())
            linear_1_res.append(Bottleneck(256, 64))
            linear_1_bn.append(nn.BatchNorm2d(256))
            linear_1_relu.append(nn.ReLU())
            linear_1_conv.append(nn.Conv2d(256, 256, padding=0, kernel_size=1,
                                           stride=1, bias=False))
            if msg_pass:
                if index == 0:
                    mp.append(MessagePassing(first=1))
                elif index == self.stacks - 1:
                    mp.append(MessagePassing(last=1))
                else:
                    mp.append(MessagePassing())
            else:
                linear_mp_bn.append(nn.BatchNorm2d(256))
                linear_mp_relu.append(nn.ReLU())
                linear_mp_conv.append(nn.Conv2d(256, 13, padding=0, kernel_size=1,
                                                stride=1, bias=False))
            if index != self.stacks - 1:
                linear_2_bn.append(nn.BatchNorm2d(256))
                linear_2_relu.append(nn.ReLU())
                linear_2_conv.append(nn.Conv2d(256, 256, padding=0, kernel_size=1,
                                     stride=1, bias=False))
                linear_3.append(nn.Conv2d(13, 256, padding=0, kernel_size=1,
                                          stride=1, bias=False))
        self.hg = nn.ModuleList(hg)
        self.linear_1_res = nn.ModuleList(linear_1_res)
        self.linear_1_bn = nn.ModuleList(linear_1_bn)
        self.linear_1_relu = nn.ModuleList(linear_1_relu)
        self.linear_1_conv = nn.ModuleList(linear_1_conv)
        self.mp = nn.ModuleList(mp)
        self.linear_2_bn = nn.ModuleList(linear_2_bn)
        self.linear_2_relu = nn.ModuleList(linear_2_relu)
        self.linear_2_conv = nn.ModuleList(linear_2_conv)
        self.linear_3 = nn.ModuleList(linear_3)
        self.linear_mp_bn = nn.ModuleList(linear_mp_bn)
        self.linear_mp_relu = nn.ModuleList(linear_mp_relu)
        self.linear_mp_conv = nn.ModuleList(linear_mp_conv)

        for m in self.modules():
            if m.__class__.__name__ in ['Conv2d']:
                nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, x):
        heatmaps = []         # save all the stacks output feature maps
        inter_level_msg = []
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.conv1_relu(out)
        out = self.pre_res_1(out)
        out = self.pool1(out)
        out = self.pre_res_2(out)
        out = self.pre_res_2_bn(out)
        out = self.pre_res_2_relu(out)
        out = self.hourglass_0(out)
        for index in range(self.stacks):
            temp = self.hg[index](out)
            temp = self.linear_1_res[index](temp)
            temp = self.linear_1_bn[index](temp)
            temp = self.linear_1_relu[index](temp)
            temp = self.linear_1_conv[index](temp)
            if self.msg_pass:
                if index != self.stacks - 1:
                    heatmap, inter_level_msg = self.mp[index](temp, inter_level_msg)
                else:
                    heatmap = self.mp[index](temp, inter_level_msg)
            else:
                heatmap = self.linear_mp_bn[index](temp)
                heatmap = self.linear_mp_relu[index](heatmap)
                heatmap = self.linear_mp_conv[index](heatmap)
            heatmaps.append(heatmap)
            if index != self.stacks - 1:
                temp = self.linear_2_bn[index](temp)
                temp = self.linear_2_relu[index](temp)
                linear2_out = self.linear_2_conv[index](temp)
                linear3_out = self.linear_3[index](heatmap)
                out = out + linear2_out + linear3_out
        return heatmaps  # 每一个stack的输出heatmap经过append