import torch
from torch import nn
import torch.nn.functional as F
from .models import Bottleneck, FMFHourglass


class Regressor(nn.Module):

    def __init__(self, classes=13, fuse_stages=4, planes=16, output=196):
        super(Regressor, self).__init__()
        self.classes = classes
        self.FMF_stages = 3
        self.fuse_stages = fuse_stages
        self.planes = planes
        self.conv1 = nn.Conv2d(14, self.planes, padding=3, kernel_size=7, stride=2, bias=False) \
            if fuse_stages > 0 else nn.Conv2d(1, self.planes, padding=3, kernel_size=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.planes)
        self.bn2 = nn.BatchNorm2d(256)  # regressor最后一个Batchnorm
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)           # regressor ip之前最后一个relu
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)  # problem, need to see the source code of caffe (solved)
        baseline_bn, baseline_relu, baseline_res_1, baseline_res_2 = [], [], [], []
        pre_fmf_bn, pre_fmf_relu, pre_fmf_conv = [], [], []
        aft_fmf_bn, aft_fmf_relu, aft_fmf_conv = [], [], []
        tanh = []
        fmfhourglass = []
        for index in range(self.FMF_stages + 1):
            if index == 0:
                baseline_bn.append(nn.BatchNorm2d(self.planes))
                baseline_relu.append(nn.ReLU())
                baseline_res_1.append(Bottleneck(self.planes, self.planes//2))
                baseline_res_2.append(Bottleneck(self.planes * 2, self.planes//2))
            else:
                baseline_bn.append(nn.BatchNorm2d(self.planes * pow(2, index)))
                baseline_relu.append(nn.ReLU())
                baseline_res_1.append(Bottleneck(self.planes * pow(2, index), self.planes * pow(2, index-1), stride=2))
                baseline_res_2.append(Bottleneck(self.planes * pow(2, index+1), self.planes * pow(2, index-1)))
        for index in range(self.FMF_stages):
            pre_fmf_bn.append(nn.BatchNorm2d(self.planes * pow(2, index+1) + self.classes))
            pre_fmf_relu.append(nn.ReLU())
            pre_fmf_conv.append(nn.Conv2d(self.planes*pow(2, index+1) + self.classes, self.planes*pow(2, index+1),
                                          padding=0, kernel_size=1, stride=1, bias=False))
        for index in range(self.FMF_stages):
            fmfhourglass.append(FMFHourglass(planes=8*pow(2, index), depth=3-index))
        for index in range(self.FMF_stages):
            aft_fmf_bn.append(nn.BatchNorm2d(self.planes * pow(2, index + 1)))
            aft_fmf_bn.append(nn.BatchNorm2d(self.planes * pow(2, index + 1)))
            aft_fmf_relu.append(nn.ReLU())
            aft_fmf_relu.append(nn.ReLU())
            aft_fmf_conv.append(nn.Conv2d(self.planes * pow(2, index + 1), self.planes * pow(2, index + 1),
                                          padding=0, kernel_size=1, stride=1, bias=False))
            aft_fmf_conv.append(nn.Conv2d(self.planes * pow(2, index + 1), self.planes * pow(2, index + 1),
                                          padding=0, kernel_size=1, stride=1, bias=False))
            tanh.append(nn.Tanh())
        self.bl_bn = nn.ModuleList(baseline_bn)
        self.bl_relu = nn.ModuleList(baseline_relu)
        self.bl_res_1 = nn.ModuleList(baseline_res_1)
        self.bl_res_2 = nn.ModuleList(baseline_res_2)
        self.pre_fmf_bn = nn.ModuleList(pre_fmf_bn)
        self.pre_fmf_relu = nn.ModuleList(pre_fmf_relu)
        self.pre_fmf_conv = nn.ModuleList(pre_fmf_conv)
        self.FMF_Hourglass = nn.ModuleList(fmfhourglass)
        self.aft_fmf_bn = nn.ModuleList(aft_fmf_bn)
        self.aft_fmf_relu = nn.ModuleList(aft_fmf_relu)
        self.aft_fmf_conv = nn.ModuleList(aft_fmf_conv)
        self.tanh = nn.ModuleList(tanh)
        self.fc1 = nn.Linear(256 * 8 * 8, 256)  # 目前的代码暂时不考虑通用性，很多数字暂时都强硬地固定下来了
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output)
        self.fc_relu1 = nn.ReLU(inplace=False)
        self.fc_relu2 = nn.ReLU(inplace=False)

        for m in self.modules():
            if m.__class__.__name__ in ['Conv2d']:
                nn.init.kaiming_uniform_(m.weight.data)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, input_img, heatmap):
        data_concat = []
        if self.fuse_stages > 0:
            out = F.interpolate(heatmap, scale_factor=4, mode='bilinear', align_corners=True)
            data_concat.append(input_img)
            for index in range(self.classes - 1):
                data_concat[0] = torch.cat((data_concat[0], input_img), 1)
            out = data_concat[0]*out
            out = torch.cat((out, input_img), 1)
        else:
            out = input_img
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.bl_bn[0](out)
        out = self.bl_relu[0](out)
        out = self.bl_res_1[0](out)
        out = self.bl_res_2[0](out)
        for index in range(self.FMF_stages):
            if index < self.fuse_stages - 1:
                temp = F.interpolate(heatmap, scale_factor=pow(2, -1*index), mode='bilinear', align_corners=True)
                temp_out = torch.cat((temp, out), 1)
                temp_out = self.pre_fmf_bn[index](temp_out)
                temp_out = self.pre_fmf_relu[index](temp_out)
                temp_out = self.pre_fmf_conv[index](temp_out)
                temp_out = self.FMF_Hourglass[index](temp_out)
                temp_out = self.aft_fmf_bn[2 * index](temp_out)
                temp_out = self.aft_fmf_relu[2 * index](temp_out)
                temp_out = self.aft_fmf_conv[2 * index](temp_out)
                temp_out = self.aft_fmf_bn[2 * index + 1](temp_out)
                temp_out = self.aft_fmf_relu[2 * index + 1](temp_out)
                temp_out = self.aft_fmf_conv[2 * index + 1](temp_out)
                temp_out = self.tanh[index](temp_out)
                temp_out = temp_out * out
                out = temp_out + out
            out = self.bl_bn[index+1](out)
            out = self.bl_relu[index + 1](out)
            out = self.bl_res_1[index + 1](out)
            out = self.bl_res_2[index + 1](out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = out.view(-1, self.num_flat_features(out))
        out = self.fc1(out)
        out = self.fc_relu1(out)
        out = self.fc2(out)
        out = self.fc_relu2(out)
        out = self.fc3(out)

        return out