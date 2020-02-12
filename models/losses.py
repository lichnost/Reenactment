import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.dataset_info import boundary_num
from utils.visual import show_img
from torchvision import models
from collections import namedtuple


class GPLoss(nn.Module):
    def __init__(self):
        super(GPLoss, self).__init__()
        self.trace = SPLoss()

    def get_image_gradients(self, input):
        f_v_1 = F.pad(input, (0, -1, 0, 0))
        f_v_2 = F.pad(input, (-1, 0, 0, 0))
        f_v = f_v_1 - f_v_2

        f_h_1 = F.pad(input, (0, 0, 0, -1))
        f_h_2 = F.pad(input, (0, 0, -1, 0))
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def __call__(self, input, reference):
        ## comment these lines when you inputs and outputs are in [0,1] range already
        # input = (input + 1) / 2
        # reference = (reference + 1) / 2

        input_v, input_h = self.get_image_gradients(input)
        ref_v, ref_h = self.get_image_gradients(reference)

        trace_v = self.trace(input_v, ref_v)
        trace_h = self.trace(input_h, ref_h)
        return trace_v + trace_h


class CPLoss(nn.Module):
    def __init__(self, rgb=True, yuv=True, yuvgrad=True):
        super(CPLoss, self).__init__()
        self.rgb = rgb
        self.yuv = yuv
        self.yuvgrad = yuvgrad
        self.trace = SPLoss()
        self.trace_YUV = SPLoss()

    def get_image_gradients(self, input):
        f_v_1 = F.pad(input, (0, -1, 0, 0))
        f_v_2 = F.pad(input, (-1, 0, 0, 0))
        f_v = f_v_1 - f_v_2

        f_h_1 = F.pad(input, (0, 0, 0, -1))
        f_h_2 = F.pad(input, (0, 0, -1, 0))
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def to_YUV(self, input):
        return torch.cat((0.299 * input[:, 0, :, :].unsqueeze(1) + 0.587 * input[:, 1, :, :].unsqueeze(
            1) + 0.114 * input[:, 2, :, :].unsqueeze(1), \
                          0.493 * (input[:, 2, :, :].unsqueeze(1) - (
                                      0.299 * input[:, 0, :, :].unsqueeze(1) + 0.587 * input[:, 1, :, :].unsqueeze(
                                  1) + 0.114 * input[:, 2, :, :].unsqueeze(1))), \
                          0.877 * (input[:, 0, :, :].unsqueeze(1) - (
                                      0.299 * input[:, 0, :, :].unsqueeze(1) + 0.587 * input[:, 1, :, :].unsqueeze(
                                  1) + 0.114 * input[:, 2, :, :].unsqueeze(1)))), dim=1)

    def __call__(self, input, reference):
        ## comment these lines when you inputs and outputs are in [0,1] range already
        # input = (input + 1) / 2
        # reference = (reference + 1) / 2

        total_loss = 0
        if self.rgb:
            total_loss += self.trace(input, reference)
        if self.yuv:
            input_yuv = self.to_YUV(input)
            reference_yuv = self.to_YUV(reference)
            total_loss += self.trace(input_yuv, reference_yuv)
        if self.yuvgrad:
            input_v, input_h = self.get_image_gradients(input_yuv)
            ref_v, ref_h = self.get_image_gradients(reference_yuv)

            total_loss += self.trace(input_v, ref_v)
            total_loss += self.trace(input_h, ref_h)

        return total_loss


class SPLoss(nn.Module):
    """
    Slow implementation of the trace loss using the same formula as stated in the paper.
    """

    def __init__(self, weight=[1., 1., 1.]):
        super(SPLoss, self).__init__()
        self.weight = weight

    def __call__(self, input, reference):
        a = 0
        b = 0
        if len(input.shape) == 3:
            for i in range(input.shape[0]):
                a += torch.trace(torch.matmul(F.normalize(input[i, :, :], p=2, dim=1), torch.t(F.normalize(reference[i, :, :], p=2, dim=1)))) / input.shape[1] * self.weight[0]
                b += torch.trace(torch.matmul(torch.t(F.normalize(input[i, :, :], p=2, dim=0)), F.normalize(reference[i, :, :], p=2, dim=0))) / input.shape[2] * self.weight[0]
        elif len(input.shape) == 4:
            for i in range(input.shape[0]):
                for j in range(input.shape[1]):
                    a += torch.trace(torch.matmul(F.normalize(input[i, j, :, :], p=2, dim=1), torch.t(F.normalize(reference[i, j, :, :], p=2, dim=1)))) / input.shape[2] * self.weight[j]
                    b += torch.trace(torch.matmul(torch.t(F.normalize(input[i, j, :, :], p=2, dim=0)),F.normalize(reference[i, j, :, :], p=2, dim=0))) / input.shape[3] * self.weight[j]

        a = -torch.sum(a) / input.shape[0]
        b = -torch.sum(b) / input.shape[0]
        return a + b


class HeatmapLoss(nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2)
        loss = loss.sum(dim=3).sum(dim=2).sum(dim=1).mean() / 2.
        return loss

class WingLoss(nn.Module):

    def __init__(self, omega=10, epsilon=2, weight=None):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)
        self.weight = weight

    def forward(self, predictions, targets):
        x = predictions - targets
        if self.weight is not None:
            x = x * self.weight
        t = torch.abs(x)

        return torch.mean(torch.where(t < self.omega, self.omega * torch.log(1 + t / self.epsilon), t - self.C))


# torch.log  and math.log is e based
class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


# feature loss network
class FeatureLoss(torch.nn.Module):
    def __init__(self, requires_grad=False, loss_type='relu2_2_and_relu3_3'):
        super(FeatureLoss, self).__init__()
        self.loss_type = loss_type
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.criterionL2 = torch.nn.MSELoss()

    def get_vgg_output(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

    def forward(self, pred, target):
        features_pred = self.get_vgg_output(pred)
        features_target = self.get_vgg_output(target)
        if self.loss_type == 'relu1_2':
            return self.criterionL2(features_pred.relu1_2,
                                                   features_target.relu1_2)
        elif self.loss_type == 'relu2_2':
            return self.criterionL2(features_pred.relu2_2,
                                                   features_target.relu2_2)
        elif self.loss_type == 'relu3_3':
            return self.criterionL2(features_pred.relu3_3,
                                                   features_target.relu3_3)
        elif self.loss_type == 'relu4_3':
            return self.criterionL2(features_pred.relu4_3,
                                                   features_target.relu4_3)
        elif self.loss_type == 'relu1_2_and_relu2_2':
            return (self.criterionL2(features_pred.relu1_2, features_target.relu1_2) +
                                   self.criterionL2(features_pred.relu2_2,
                                                    features_target.relu2_2))
        elif self.loss_type == 'relu2_2_and_relu3_3':
            return (self.criterionL2(features_pred.relu2_2, features_target.relu2_2) +
                                   self.criterionL2(features_pred.relu3_3,
                                                    features_target.relu3_3))
        elif self.loss_type == 'relu3_3_and_relu4_3':
            return (self.criterionL2(features_pred.relu3_3, features_target.relu3_3) +
                                   self.criterionL2(features_pred.relu4_3,
                                                    features_target.relu4_3))