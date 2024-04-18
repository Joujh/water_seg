import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

from mmseg.models import LOSSES


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class MS_SSIM(torch.nn.Module):
    def __init__(self, size_average=True, max_val=255):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val

    def _ssim(self, img1, img2, size_average=True):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).cuda()
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=self.channel) - mu1_mu2

        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())

        msssim = Variable(torch.Tensor(levels, ).cuda())
        mcs = Variable(torch.Tensor(levels, ).cuda())
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = torch.abs((torch.prod(mcs[0:levels - 1] ** weight[0:levels - 1]) *
                           (msssim[levels - 1] ** weight[levels - 1])))
        return value

    def forward(self, img1, img2):

        return self.ms_ssim(img1, img2)


def color_loss(y_true, y_pred):
    return torch.mean(torch.abs(torch.mean(y_true, dim=[2, 3]) - torch.mean(y_pred, dim=[2, 3])))


def maxcentropy_loss(y_true, y_pred):
    y_pred = torch.max(y_pred, dim=3, keepdim=True)[0]
    return torch.mean(torch.abs(y_pred - y_true))


# def psnr_loss(y_true, y_pred):
#     mse = torch.mean((y_true - y_pred) ** 2)
#     psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
#     return 40.0 - psnr


def histogram_loss(y_true, y_pred, bins=256):
    # 将数据扁平化
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # 计算直方图
    y_true_hist = torch.histc(y_true_flat.float(), bins=bins, min=0, max=1)
    y_pred_hist = torch.histc(y_pred_flat.float(), bins=bins, min=0, max=1)

    # 归一化直方图
    y_true_hist = y_true_hist.float() / y_true_hist.sum()
    y_pred_hist = y_pred_hist.float() / y_pred_hist.sum()

    # 计算直方图的L1距离
    hist_distance = torch.mean(torch.abs(y_true_hist - y_pred_hist))

    return hist_distance


def lyt_loss(y_true, y_pred):
    # y_true = (y_true + 1.0) / 2.0
    # y_pred = (y_pred + 1.0) / 2.0
    multiscale_ssim_loss = MS_SSIM()

    ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
    maxcentropy_l = maxcentropy_loss(y_true, y_pred)
    hist_l = histogram_loss(y_true, y_pred)
    # psnr_l = psnr_loss(y_true, y_pred)
    color_l = color_loss(y_true, y_pred)

    total_loss = ms_ssim_l * 0.05 + maxcentropy_l * 0.1 + hist_l * 0.05 + color_l * 0.1
    total_loss = torch.mean(torch.abs(total_loss))
    return total_loss


@LOSSES.register_module()
class LytLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.name = "LytLoss"
        self.loss_weight = loss_weight

    def forward(self, x, y):
        return lyt_loss(x, y) * self.loss_weight
