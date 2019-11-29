import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.dataset_info import boundary_num
from utils.train_eval_utils import get_heatmap_gray
from utils.visual import show_img


class SPLFullLoss(nn.Module):
    """
    Slow implementation of the trace loss using the same formula as stated in the paper.
    """
    def __init__(self,weight = 1.):
        super(SPLFullLoss, self).__init__()
        self.weight = weight

    def __call__(self, input, reference):
        a = 0
        b = 0
        for i in range(input.shape[0]):
            a += torch.trace(torch.matmul(F.normalize(input[i,:,:],p=2,dim=1),torch.t(F.normalize(reference[i,:,:],p=2,dim=1))))/input.shape[1]*self.weight
            b += torch.trace(torch.matmul(torch.t(F.normalize(input[i,:,:],p=2,dim=0)),F.normalize(reference[i,:,:],p=2,dim=0)))/input.shape[2]*self.weight
        a = -torch.sum(a)/input.shape[0]
        b = -torch.sum(b)/input.shape[0]
        return a+b


class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def __call__(self, input, reference):
        a = torch.sum(
            torch.sum(F.normalize(input, p=2, dim=1) * F.normalize(reference, p=2, dim=1), dim=1, keepdim=True))
        b = torch.sum(
            torch.sum(F.normalize(input, p=2, dim=2) * F.normalize(reference, p=2, dim=2), dim=2, keepdim=True))
        return -(a + b) / input.size(2)


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


    def __call__(self, pred, gt):
        ## comment these lines when you inputs and outputs are in [0,1] range already
        # pred = (pred + 1) / 2
        # gt = (gt + 1) / 2
        pred = get_heatmap_gray(pred)
        gt = get_heatmap_gray(gt)

        input_v, input_h = self.get_image_gradients(pred)
        ref_v, ref_h = self.get_image_gradients(gt)

        trace_v = self.trace(input_v, ref_v)
        trace_h = self.trace(input_h, ref_h)
        return trace_v + trace_h


class GPFullLoss(nn.Module):
    def __init__(self):
        super(GPFullLoss, self).__init__()
        self.trace = SPLFullLoss()

    def get_image_gradients(self, input):
        f_v_1 = F.pad(input, (0, -1, 0, 0))
        f_v_2 = F.pad(input, (-1, 0, 0, 0))
        f_v = f_v_1 - f_v_2

        f_h_1 = F.pad(input, (0, 0, 0, -1))
        f_h_2 = F.pad(input, (0, 0, -1, 0))
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def __call__(self, pred, gt):
        ## comment these lines when you inputs and outputs are in [0,1] range already
        # pred = (pred + 1) / 2
        # gt = (gt + 1) / 2
        pred = get_heatmap_gray(pred)
        gt = get_heatmap_gray(gt)

        input_v, input_h = self.get_image_gradients(pred)
        ref_v, ref_h = self.get_image_gradients(gt)

        trace_v = self.trace(input_v, ref_v)
        trace_h = self.trace(input_h, ref_h)
        return trace_v + trace_h


class HeatmapLoss(nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2)
        loss = loss.sum(dim=3).sum(dim=2).sum(dim=1).mean() / 2.
        return loss


class WingLoss(nn.Module):

    def __init__(self, w=10, epsilon=2, weight=None):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * np.log(1 + self.w / self.epsilon)
        self.weight = weight

    def forward(self, predictions, targets):
        x = predictions - targets
        if self.weight is not None:
            x = x * self.weight
        t = torch.abs(x)

        return torch.mean(torch.where(t < self.w, self.w * torch.log(1 + t / self.epsilon), t - self.C))
