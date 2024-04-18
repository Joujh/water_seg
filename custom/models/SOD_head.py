import torch
import torch.nn as nn
from torch import Tensor
from mmseg.models.builder import LOSSES
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from .utils import Representations, ResidualBlock, LightAttention
from mmseg.ops import resize
from typing import Tuple
from mmseg.models.builder import HEADS



import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class AdaDConv(nn.Module):
    """
    Adaptive-weighted downsampling
    """

    def __init__(self, in_channels, kernel_size=3, stride=2, groups=1, use_channel=True, use_nin=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.stride = stride
        self.in_channels = in_channels
        self.groups = groups
        self.use_channel = use_channel

        if use_nin:
            mid_channel = min((kernel_size ** 2 // 2), 4)
            self.weight_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=groups * mid_channel, stride=stride,
                          kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
                nn.BatchNorm2d(self.groups * mid_channel),
                nn.ReLU(True),
                nn.Conv2d(in_channels=groups * mid_channel, out_channels=groups * kernel_size ** 2, stride=1,
                          kernel_size=1, bias=False, padding=0, groups=groups),
                nn.BatchNorm2d(self.groups * kernel_size ** 2),
            )

        else:
            self.weight_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride,
                          kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
                nn.BatchNorm2d(self.groups * kernel_size ** 2),
                # nn.Softmax(dim=1)
            )

        if use_channel:
            self.channel_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1, bias=False),
                # nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels // 4, out_channels=in_channels, kernel_size=1, bias=False),
                # nn.Sigmoid()
            )

        # nn.init.kaiming_normal_(self.channel_net[0].weight, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.weight_net[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, c, h, w = x.shape
        oh = (h - 1) // self.stride + 1
        ow = (w - 1) // self.stride + 1
        weight = self.weight_net(x)
        _weight = weight
        weight = weight.reshape(b, self.groups, 1, self.kernel_size ** 2, oh, ow)
        weight = weight.repeat(1, 1, c // self.groups, 1, 1, 1)

        if self.use_channel:
            tmp = self.channel_net(x).reshape(b, self.groups, c // self.groups, 1, 1, 1)
            # tmp[tmp < 1.] = tmp[tmp < 1.] ** 2
            # print(weight.shape)
            weight = weight * tmp
        weight = weight.permute(0, 1, 2, 4, 5, 3).softmax(dim=-1)
        weight = weight.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)

        pad_x = F.pad(x, pad=[self.pad] * 4, mode='reflect')
        # shape:  B x C x H // stride x W //stride x ksize x ksize
        pad_x = pad_x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        pad_x = pad_x.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
        res = weight * pad_x
        res = res.sum(dim=(-1, -2)).reshape(b, c, oh, ow)
        return res


@HEADS.register_module()
class SODHead(BaseModule):
    def __init__(
            self,
            channels,
            in_channels=3,
            base_channels=32,
            num_downsample=2,
            num_resblock=2,
            attn_channels=16,
            image_pool_channels=32,
            ill_embeds_op="+",
            clip=True,
            gray_illumination=False,
            eps=1e-5,
            loss_retinex=None,
            loss_smooth=None,
            loss_lyt=None,
            conv_cfg=None,
            norm_cfg=dict(type="IN2d"),
            act_cfg=dict(type="ReLU"),
            align_corners=False,
            init_cfg=dict(type="Normal", std=0.01),
    ):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.fp16_enabled = False
        self.loss_retinex = LOSSES.build(loss_retinex)
        self.loss_smooth = LOSSES.build(loss_smooth)
        self.loss_lyt = LOSSES.build(loss_lyt)
        self.eps = eps
        self.init_autoencoder(
            base_channels,
            num_downsample,
            num_resblock,
            attn_channels,
            image_pool_channels,
        )
        self.reflectance_output = nn.Sequential(
            nn.Conv2d(
                self.channels,
                3,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
            nn.Tanh(),
        )
        self.illumination_output = nn.Sequential(
            nn.Conv2d(
                self.channels,
                3,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
            nn.Sigmoid(),
        )
        self.ill_embeds_op = ill_embeds_op
        self.clip = clip
        self.gray_illumination = gray_illumination

    def init_autoencoder(
            self,
            base_channels,
            num_downsample,
            num_resblock,
            attn_channels,
            image_pool_channels,
    ):
        assert (
                num_resblock >= 1
                and num_downsample >= 1
                and attn_channels >= 1
                and image_pool_channels >= 1
        )
        channels = base_channels
        self.stem = ConvModule(
            self.in_channels,
            channels,
            kernel_size=7,
            padding=3,
            padding_mode="reflect",
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        down_layers = []
        for _ in range(num_downsample):
            down_layers += [
                ConvModule(
                    channels,
                    channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    padding_mode="reflect",
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            ]
            # down_layers += [
            #     nn.Sequential(
            #         AdaDConv(channels),
            #         nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=3, stride=1, padding=1)
            #     )
            # ]
            channels *= 2
        self.downsample = nn.Sequential(*down_layers)
        res_layers = []
        for _ in range(num_resblock):
            res_layers += [
                ResidualBlock(
                    channels,
                    self.conv_cfg,
                    self.act_cfg,
                    self.norm_cfg,
                    "reflect",
                )
            ]
        self.residual = nn.Sequential(*res_layers)
        self.light_attention = LightAttention(channels, attn_channels, pre_downsample=2)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, image_pool_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.merge = ConvModule(
            channels + attn_channels + image_pool_channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        upsample_ill_layers, upsample_ref_layers = [], []
        for _ in range(num_downsample):
            upsample_ill_layers += [
                ConvModule(
                    channels,
                    channels // 2,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=self.align_corners
                ),
            ]
            upsample_ref_layers += [
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=self.align_corners
                ),
                ConvModule(
                    channels,
                    channels // 2,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
            ]
            channels //= 2
        upsample_ref_layers += [nn.Conv2d(channels, channels, kernel_size=1)]
        self.upsample_illumination = nn.Sequential(*upsample_ill_layers)
        self.upsample_reflectance = nn.Sequential(*upsample_ref_layers)
        self.refine_reflectance = ConvModule(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def _forward_feature(self, imgs: Tensor) -> Tensor:
        img_embeds = self.stem(imgs)
        feats = [self.downsample(img_embeds)]
        feats += [
            resize(
                self.image_pool(feats[0]),
                size=feats[0].shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            ),
            self.light_attention(feats[0]),
        ]
        feats = torch.cat(feats, dim=1)
        feats = self.merge(feats)
        feats = self.residual(feats)
        ill_embeds = self.upsample_illumination(feats)
        ref_embeds = self.upsample_reflectance(feats)
        if self.ill_embeds_op == "+":
            ref_embeds = self.refine_reflectance(ref_embeds + ill_embeds + img_embeds)
        elif self.ill_embeds_op == "-":
            ref_embeds = self.refine_reflectance(ref_embeds - ill_embeds + img_embeds)
        return ref_embeds, ill_embeds, feats

    @auto_fp16(apply_to=("imgs",))
    def forward(self, imgs: Tensor) -> Representations:
        ref_embeds, ill_embeds, feats = self._forward_feature(
            torch.cat([imgs, torch.max(imgs, dim=1, keepdim=True).values], dim=1)
        )
        illumination = self.illumination_output(ill_embeds)
        illumination = torch.mean(illumination, dim=1, keepdim=True).repeat(1, 3, 1, 1)
        reflectance = self.reflectance_output(ref_embeds) + imgs
        return Representations(illumination, reflectance, feats, clip=self.clip)

    def forward_train(self, imgs: Tensor) -> Tuple[Representations, dict]:
        repres = self.forward(imgs)
        losses = dict(
            loss_smooth=self.loss_smooth(repres.illumination, repres.reflectance),
            loss_retinex=self.loss_retinex(repres.retinex(), imgs))
            #loss_lyt=self.loss_lyt(imgs, repres.retinex()))
        return repres, losses
