import os
import glob
from PIL import Image
import numpy as np
import os, glob
from torchvision.utils import save_image
import torchvision
import fvcore.common.config
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce
from denoising_diffusion_pytorch.efficientnet import efficientnet_b7, EfficientNet_B7_Weights
from denoising_diffusion_pytorch.resnet import resnet101, ResNet101_Weights
from denoising_diffusion_pytorch.swin_transformer import swin_b, Swin_B_Weights
from denoising_diffusion_pytorch.vgg import vgg16, VGG16_Weights


# from denoising_diffusion_pytorch.wcc import fft
### Compared to unet4:
# 1. add FFT-Conv on the mid feature.
######## Attention Layer ##########

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        # self.class_token_pos = nn.Parameter(torch.zeros(1, 1, num_pos_feats * 2))
        # self.class_token_pos

    def forward(self, x):
        # x: b, h, w, d
        num_feats = x.shape[3]
        num_pos_feats = num_feats // 2
        # mask = tensor_list.mask
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[2], device=x.device).to(torch.bool)
        batch = mask.shape[0]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-5
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # pos = torch.cat((pos_y, pos_x), dim=3).flatten(1, 2)
        pos = torch.cat((pos_y, pos_x), dim=3).contiguous()
        '''
        pos_x: b ,h, w, d//2
        pos_y: b, h, w, d//2
        pos: b, h, w, d
        '''
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, feature_size, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return torch.cat([x, pos], dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAtt(nn.Module):
    def __init__(self, in_dim):
        super(SpatialAtt, self).__init__()
        self.map = nn.Conv2d(in_dim, 1, 1)
        self.q_conv = nn.Conv2d(1, 1, 1)
        self.k_conv = nn.Conv2d(1, 1, 1)
        self.activation = nn.Softsign()

    def forward(self, x):
        b, _, h, w = x.shape
        att = self.map(x)  # b, 1, h, w
        q = self.q_conv(att)  # b, 1, h, w
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = self.k_conv(att)
        k = rearrange(k, 'b c h w -> b c (h w)')
        att = rearrange(att, 'b c h w -> b (h w) c')
        att = F.softmax(q @ k, dim=-1) @ att  # b, hw, 1
        att = att.reshape(b, 1, h, w)
        return self.activation(att) * x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BasicAttetnionLayer(nn.Module):
    def __init__(self, embed_dim=128, nhead=8, ffn_dim=512, window_size1=[4, 4],
                 window_size2=[1, 1], dropout=0.1):
        super().__init__()
        self.window_size1 = window_size1
        self.window_size2 = window_size2
        self.avgpool_q = nn.AvgPool2d(kernel_size=window_size1)
        self.avgpool_k = nn.AvgPool2d(kernel_size=window_size2)
        self.softmax = nn.Softmax(dim=-1)
        self.nhead = nhead

        self.q_lin = nn.Linear(embed_dim, embed_dim)
        self.k_lin = nn.Linear(embed_dim, embed_dim)
        self.v_lin = nn.Linear(embed_dim, embed_dim)

        self.mlp = Mlp(in_features=embed_dim, hidden_features=ffn_dim, drop=dropout)
        self.pos_enc = PositionEmbeddingSine(embed_dim)
        self.concat_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1)
        self.gn = nn.GroupNorm(8, embed_dim)

        self.out_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x1, x2):  # x1 for q (conditional input), x2 for k,v
        B, C1, H1, W1 = x1.shape
        _, C2, H2, W2 = x2.shape
        # x1 = x1.permute(0, 2, 3, 1).contiguous() # B, H1, W1, C1
        shortcut = x2 + self.concat_conv(torch.cat(
            [F.interpolate(x1, size=(H2, W2), mode='bilinear', align_corners=True),
             x2], dim=1))
        shortcut = self.gn(shortcut)
        pad_l = pad_t = 0
        pad_r = (self.window_size1[1] - W1 % self.window_size1[1]) % self.window_size1[1]
        pad_b = (self.window_size1[0] - H1 % self.window_size1[0]) % self.window_size1[0]
        x1 = F.pad(x1, (pad_l, pad_r, pad_t, pad_b, 0, 0))
        _, _, H1p, W1p = x1.shape
        # x2 = x2.permute(0, 2, 3, 1).contiguous()  # B, H2, W2, C2
        pad_l = pad_t = 0
        pad_r = (self.window_size2[1] - W2 % self.window_size2[1]) % self.window_size2[1]
        pad_b = (self.window_size2[0] - H2 % self.window_size2[0]) % self.window_size2[0]
        x2 = F.pad(x2, (pad_l, pad_r, pad_t, pad_b, 0, 0))
        _, _, H2p, W2p = x2.shape
        # x1g = x1 #B, C1, H1p, W1p
        # x2g = x2 #B, C2, H2p, W2p
        x1_s = self.avgpool_q(x1)
        qg = self.avgpool_q(x1).permute(0, 2, 3, 1).contiguous()
        qg = qg + self.pos_enc(qg)
        qg = qg.view(B, -1, C2)
        kg = self.avgpool_k(x2).permute(0, 2, 3, 1).contiguous()
        kg = kg + self.pos_enc(kg)
        kg = kg.view(B, -1, C1)
        num_window_q = qg.shape[1]
        num_window_k = kg.shape[1]
        qg = self.q_lin(qg).reshape(B, num_window_q, self.nhead, C1 // self.nhead).permute(0, 2, 1,
                                                                                           3).contiguous()
        kg2 = self.k_lin(kg).reshape(B, num_window_k, self.nhead, C1 // self.nhead).permute(0, 2, 1,
                                                                                            3).contiguous()
        vg = self.v_lin(kg).reshape(B, num_window_k, self.nhead, C1 // self.nhead).permute(0, 2, 1,
                                                                                           3).contiguous()
        kg = kg2
        attn = (qg @ kg.transpose(-2, -1))
        attn = self.softmax(attn)
        qg = (attn @ vg).transpose(1, 2).reshape(B, num_window_q, C1)
        qg = qg.transpose(1, 2).reshape(B, C1, H1p // self.window_size1[0], W1p // self.window_size1[1])
        # qg = F.interpolate(qg, size=(H1p, W1p), mode='bilinear', align_corners=False)
        x1_s = x1_s + qg
        x1_s = x1_s + self.mlp(x1_s)
        x1_s = F.interpolate(x1_s, size=(H2, W2), mode='bilinear', align_corners=True)
        x1_s = shortcut + self.out_conv(x1_s)
        # x1_s = self.out_norm(x1_s)
        return x1_s


class RelationNet(nn.Module):
    def __init__(self, in_channel1=128, in_channel2=128, nhead=8, layers=3, embed_dim=128, ffn_dim=512,
                 window_size1=[4, 4], window_size2=[1, 1]):
        # self.attention = BasicAttetnionLayer(embed_dim=embed_dim, nhead=nhead, ffn_dim=ffn_dim,
        #                                      window_size1=window_size1, window_size2=window_size2, dropout=0.1)
        super().__init__()
        self.layers = layers
        self.input_conv1 = nn.Sequential(
            nn.Conv2d(in_channel1, embed_dim, 1),
            nn.BatchNorm2d(embed_dim, momentum=0.03, eps=0.001),
        )
        self.input_conv2 = nn.Sequential(
            nn.Conv2d(in_channel2, embed_dim, 1),
            nn.BatchNorm2d(embed_dim, momentum=0.03, eps=0.001),
        )
        # self.input_conv1 = ConvModule(in_channel1,
        #                               embed_dim,
        #                               1,
        #                               norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        #                               act_cfg=None)
        # self.input_conv2 = ConvModule(in_channel2,
        #                               embed_dim,
        #                               1,
        #                               norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        #                               act_cfg=None)
        # self.input_conv2 = nn.Linear(in_channel2, embed_dim)
        self.attentions = nn.ModuleList()
        for i in range(layers):
            self.attentions.append(
                BasicAttetnionLayer(embed_dim=embed_dim, nhead=nhead, ffn_dim=ffn_dim,
                                    window_size1=window_size1, window_size2=window_size2, dropout=0.1)
            )

    def forward(self, cond, feat):
        # cluster = cluster.unsqueeze(0).repeat(feature.shape[0], 1, 1, 1)
        cond = self.input_conv1(cond)
        feat = self.input_conv2(feat)
        for att in self.attentions:
            feat = att(cond, feat)
        return feat


################# U-Net model defenition ####################

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class BlockFFT(nn.Module):
    def __init__(self, dim, h, w, groups=8):
        super().__init__()
        # self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02)
        # self.complex_weight = nn.Parameter(torch.normal(mean=0, std=0.01, size=(dim, h, w // 2 + 1, 2), dtype=torch.float32))
        # self.norm = nn.GroupNorm(groups, dim)
        # self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        B, C, H, W = x.shape
        # print("Input shape:", x.shape)#1,512,10,10
        # print("Complex weight shape:", self.complex_weight.shape)#512,10,6,2
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        x = x * torch.view_as_complex(self.complex_weight)
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        x = x.reshape(B, C, H, W)

        return x


class BlockGabor(nn.Module):
    def __init__(self, dim, h, w, kernel_size=15, sigma=5.0, lambd=10.0, gamma=0.5):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma

        # Generate Gabor filters for every 15 degrees
        self.theta_list = torch.arange(0, math.pi, math.pi / 12)  # 0 to 180 degrees, 15 degrees per step
        self.num_filters = len(self.theta_list)  # Number of directions

        # Initialize Gabor filters
        self.gabor_filters = self._create_gabor_filters()
        self.adjust_conv = nn.Conv2d(dim * self.num_filters, dim, kernel_size=1)

    def _create_gabor_filters(self):
        """
        Create a set of Gabor filters for each direction.
        """
        filters = []
        for theta in self.theta_list:
            gabor_kernel = self._gabor_kernel(theta)
            filters.append(gabor_kernel)
        filters = torch.stack(filters, dim=0)  # Shape: (num_filters, 1, kernel_size, kernel_size)
        filters = filters.squeeze(2)
        filters = filters.unsqueeze(0)  # Shape: (1, num_filters, 1, kernel_size, kernel_size)
        filters = filters.repeat(self.dim, 1, 1, 1, 1)  # Shape: (dim, num_filters, 1, kernel_size, kernel_size)
        filters = filters.view(self.dim * self.num_filters, 1, self.kernel_size,
                               self.kernel_size)  # Shape: (dim * num_filters, 1, kernel_size, kernel_size)
        return nn.Parameter(filters, requires_grad=False)

    def _gabor_kernel(self, theta):
        """
        Generate a single Gabor kernel for a given theta (direction).
        """
        kernel_size = self.kernel_size
        sigma = self.sigma
        lambd = self.lambd
        gamma = self.gamma

        # Create grid
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        y, x = torch.meshgrid(y, x)

        # Rotate coordinates
        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

        # Gabor formula
        gabor = torch.exp(-(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2 * sigma ** 2)) * torch.cos(
            2 * math.pi * x_theta / lambd)
        gabor = gabor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kernel_size, kernel_size)
        return gabor

    def forward(self, x):
        B, C, H, W = x.shape
        # Apply Gabor filters
        x = x.unsqueeze(1)  # Shape: (B, 1, C, H, W)
        x = F.conv2d(x.view(B * 1, C, H, W), self.gabor_filters, padding=self.kernel_size // 2, groups=self.dim)
        x = x.view(B, self.num_filters, C, H, W)  # Shape: (B, num_filters, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # Shape: (B, C, num_filters, H, W)
        x = x.reshape(B, C * self.num_filters, H, W)  # Shape: (B, C * num_filters, H, W)
        x = self.adjust_conv(x)
        return x


class EnhancedBlockGabor(nn.Module):
    def __init__(self, dim, h, w, kernel_size=15, scales=3, orientations=8,
                 sigma=5.0, lambd=10.0, gamma=0.5):
        super().__init__()
        self.dim = dim
        self.scales = scales
        self.orientations = orientations
        self.kernel_size = kernel_size

        self.frequencies = [1.0 / (2 ** s) for s in range(scales)]  # 1.0, 0.5, 0.25
        self.register_buffer('filters', self._create_complex_filters(sigma, lambd, gamma))
        self.adjust_conv = nn.Conv2d(dim * scales * orientations, dim, 1)
        self.norm = nn.InstanceNorm2d(dim)

    def _create_complex_filters(self, sigma, lambd, gamma):
        """生成多尺度复数Gabor滤波器组"""
        filters = []
        coords = torch.arange(-(self.kernel_size // 2), (self.kernel_size // 2) + 1)
        y, x = torch.meshgrid(coords, coords)

        for scale in range(self.scales):
            fu = self.frequencies[scale]
            alpha = fu / gamma
            beta = fu / (gamma * math.sqrt(2))

            for orient in range(self.orientations):
                theta = orient * math.pi / self.orientations
                x_theta = x * math.cos(theta) + y * math.sin(theta)
                y_theta = -x * math.sin(theta) + y * math.cos(theta)
                envelope = torch.exp(-(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2 * sigma ** 2))
                carrier = torch.exp(1j * 2 * math.pi * fu * x_theta)
                kernel = (fu ** 2 / (math.pi * gamma ** 2)) * envelope * carrier
                filters.append(torch.stack([kernel.real, kernel.imag]))  # 实部+虚部

        return torch.stack(filters)  # (scales*orientations, 2, kernel_size, kernel_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(1, -1, H, W)  # 合并批次和通道
        filters = self.filters.repeat_interleave(C, dim=0)  # (C*S*O, 2, K, K)
        filters = filters.view(-1, 1, self.kernel_size, self.kernel_size)  # (2*C*S*O, 1, K, K)
        features = F.conv2d(x, filters, padding=self.kernel_size // 2,
                            groups=B * C * self.scales * self.orientations * 2)
        features = features.view(B, C, self.scales, self.orientations, 2, H, W)
        # 计算幅度特征
        mag = torch.sqrt(features[..., 0, :, :] ** 2 + features[..., 1, :, :] ** 2)  # (B, C, S, O, H, W)
        mag = mag.permute(0, 1, 2, 3, 5, 4).reshape(B, -1, H, W)  # (B, C*S*O, H, W)
        return self.norm(self.adjust_conv(mag))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class ResnetBlockFFT(nn.Module):
    def __init__(self, dim, dim_out, h, w, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = BlockFFT(dim_out, h, w, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class ResnetDownsampleBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = nn.Sequential(
            WeightStandardizedConv2d(dim_out, dim_out, 3, stride=2, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(
            F.interpolate(x, size=h.shape[-2:], mode="bilinear")
        )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class ColorOpponency(nn.Module):
    def __init__(self):
        super(ColorOpponency, self).__init__()
        weight = torch.tensor([
            [0.299, 0.587, 0.114],    # Luminance (亮度)
            [0.5, -0.5, 0],           # Red-Green (红-绿)
            [-0.169, -0.331, 0.5]     # Blue-Yellow (蓝-黄)
        ], dtype=torch.float32).view(3, 3, 1, 1)
        # self.weight = self.weight.unsqueeze(-1).unsqueeze(-1)  # 扩展为卷积核形状 (3, 3, 1, 1)
        # self.bias = torch.zeros(3)  # 偏置设为0
        self.register_buffer('weight', weight)
        self.register_buffer('bias', torch.zeros(3))

    def forward(self, x):
        # x 的形状: (batch_size, 3, H, W)
        # 使用 1x1 卷积实现线性变换
        # return F.conv2d(x, self.weight.to(x.device), self.bias.to(x.device))
        # 输入: x.shape = [B, 3, H, W]
        # 输出: [B, 3, H, W]，分别是亮度、红绿通道、蓝黄通道

        # 应用 1x1 卷积进行颜色拮抗变换
        y = F.conv2d(x, self.weight, self.bias)
        #全图归一化 training
        # mean = y.mean(dim=(0, 2, 3), keepdim=True)
        # std = y.std(dim=(0, 2, 3), keepdim=True) + 1e-6
        mean = y.mean(dim=(2, 3), keepdim=True)
        std = y.std(dim=(2, 3), keepdim=True) + 1e-6
        y = (y - mean) / std  # 标准化到零均值、单位方差

        #归一化 test
        # luminance = y[:, 0:1, :, :]
        # # mean = y.mean(dim=(2, 3), keepdim=True)
        # # std = y.std(dim=(2, 3), keepdim=True) + 1e-6  # 避免除以0
        # mean = luminance.mean(dim=(2, 3), keepdim=True)
        # std = luminance.std(dim=(2, 3), keepdim=True) + 1e-6
        # luminance_norm = (luminance - mean) / std
        # alpha = 0.6  # 保留一些原始局部特征
        # luminance_blend = alpha * luminance_norm + (1 - alpha) * (luminance - mean)
        #
        # # 保留对比通道
        # rg = y[:, 1:2, :, :]
        # by = y[:, 2:3, :, :]
        #
        # y = torch.cat([luminance_norm, rg, by], dim=1)
        # y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)#空间平滑
        return y

# class ColorOpponency(nn.Module):
#     def __init__(self):
#         super(ColorOpponency, self).__init__()
#         # 定义颜色拮抗变换的权重矩阵
#         self.weight = torch.tensor([
#             [0.299, 0.587, 0.114],  # Luminance (亮度)
#             [0.5, -0.5, 0],  # Red-Green (红-绿)
#             [-0.169, -0.331, 0.5]  # Blue-Yellow (蓝-黄)
#         ], dtype=torch.float32)
#         self.weight = self.weight.unsqueeze(-1).unsqueeze(-1)  # 扩展为卷积核形状 (3, 3, 1, 1)
#         self.bias = torch.zeros(3)  # 偏置设为0
#
#     def forward(self, x):
#         # x 的形状: (batch_size, 3, H, W)
#         # 使用 1x1 卷积实现线性变换
#         y = F.conv2d(x, self.weight.to(x.device), self.bias.to(x.device))
#         return y


class TextureSuppressionLayer(nn.Module):
    def __init__(self, nThetas=8):
        super().__init__()
        self.nThetas = nThetas
        self.register_buffer('_dummy', torch.tensor(0.))

        self.sigmax = 3.65
        self.sigmay = self.sigmax / 8
        self.surround_scale = 3

        self._precompute_kernels()

    def _precompute_kernels(self):
        self.center_kernels = nn.ParameterList()
        self.surround_kernels = nn.ParameterList()

        for t in range(self.nThetas):
            theta = torch.tensor(t * torch.pi / self.nThetas, device=self.device)

            center_k = self._create_gaussian_kernel(self.sigmax, self.sigmay, theta).view(1, 1, 21, 21)
            surround_k = self._create_gaussian_kernel(
                self.sigmax * self.surround_scale,
                self.sigmay * self.surround_scale,
                theta
            ).view(1, 1, 21, 21)

            self.center_kernels.append(nn.Parameter(center_k, requires_grad=False))
            self.surround_kernels.append(nn.Parameter(surround_k, requires_grad=False))

    def _create_gaussian_kernel(self, sigma_x, sigma_y, theta):
        size = 21
        half = size // 2
        x = torch.arange(-half, half + 1, device=self.device)
        y = torch.arange(-half, half + 1, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        xx_rot = xx * cos_theta + yy * sin_theta
        yy_rot = -xx * sin_theta + yy * cos_theta

        exponent = -0.5 * (xx_rot.square() / sigma_x ** 2 + yy_rot.square() / sigma_y ** 2)
        kernel = torch.exp(exponent)
        return kernel / kernel.sum()

    def forward(self, x):
        if isinstance(x, list):
            return [self._process_single_tensor(item) for item in x]
        return self._process_single_tensor(x)

    def _process_single_tensor(self, x):
        B, C, H, W = x.shape
        result = torch.empty_like(x)

        for c in range(C):
            channel_data = x[:, c:c + 1]

            with torch.no_grad():
                multi_orient = channel_data.expand(-1, -1, -1, -1).unsqueeze(-1).repeat(1, 1, 1, 1, self.nThetas)
                max_vals = multi_orient.max(dim=1)[0]
                avg_response = max_vals.mean(dim=-1)

            suppressed = self._suppress_channel(avg_response, channel_data)
            result[:, c:c + 1] = suppressed

        return result

    def _suppress_channel(self, avg_response, channel_data):
        B, H, W = avg_response.shape

        # 动态计算局部标准差
        with torch.no_grad():
            pool_size = min(3, H // 4)
            if pool_size % 2 == 0: pool_size -= 1

            avg_sq = F.avg_pool2d(avg_response.unsqueeze(1).square(), kernel_size=pool_size,
                                  padding=pool_size // 2, stride=1)
            avg = F.avg_pool2d(avg_response.unsqueeze(1), kernel_size=pool_size,
                               padding=pool_size // 2, stride=1)
            local_std = torch.sqrt(torch.clamp(avg_sq - avg.square(), min=1e-6))
            local_std = (local_std - local_std.min()) / (local_std.max() - local_std.min() + 1e-8)

        suppressed = torch.zeros_like(avg_response)

        avg_response_input = avg_response.unsqueeze(1)  # [B,1,H,W]

        for t in range(self.nThetas):
            center_k = self.center_kernels[t].repeat(B, 1, 1, 1)
            surround_k = self.surround_kernels[t].repeat(B, 1, 1, 1)

            center_resp = F.conv2d(avg_response_input, center_k, padding='same', groups=B).squeeze(1)
            surround_resp = F.conv2d(avg_response_input, surround_k, padding='same', groups=B).squeeze(1)

            std_adjusted = local_std.squeeze(1)
            suppression = torch.clamp(center_resp - std_adjusted * surround_resp, min=0)

            # 简化方向掩码逻辑（用平均分配方式代替 argmax）
            mask = (torch.arange(B, device=avg_response.device) % self.nThetas == t).view(B, 1, 1)
            suppressed = torch.where(mask, suppression, suppressed)

        return suppressed.unsqueeze(1)

    @property
    def device(self):
        return self._dummy.device


class ConditionEncoder(nn.Module):
    # CNN Resnet下采样
    def __init__(self,
                 down_dim_mults=(2, 4, 8),
                 dim=64,
                 in_dim=1,
                 out_dim=64):
        super(ConditionEncoder, self).__init__()

        self.color_opponency = ColorOpponency()
        self.color_adjust = nn.Conv2d(3, in_dim, kernel_size=1)

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=min(dim // 4, 8), num_channels=dim),
        )
        # 下采样
        self.num_resolutions = len(down_dim_mults)
        self.downs = nn.ModuleList()
        in_mults = (1,) + tuple(down_dim_mults[:-1])
        in_dims = [mult * dim for mult in in_mults]
        out_dims = [mult * dim for mult in down_dim_mults]
        for i_level in range(self.num_resolutions):
            block_in = in_dims[i_level]
            block_out = out_dims[i_level]
            self.downs.append(ResnetDownsampleBlock(dim=block_in,
                                                    dim_out=block_out))
        # 输出卷积层
        if self.num_resolutions < 1:
            self.out_conv = nn.Conv2d(dim, out_dim, 1)
        else:
            self.out_conv = nn.Conv2d(out_dims[-1], out_dim, 1)

    def forward(self, x):
        # 输入 x 的形状: (batch_size, 3, H, W)
        # 颜色拮抗变换
        x_color = self.color_opponency(x)  # 输出形状: (batch_size, 3, H, W)
        x_color = self.color_adjust(x_color)  # 输出形状: (batch_size, in_dim, H, W)
        x = self.init_conv(x_color)  # 输出形状: (batch_size, dim, H, W)
        # x = self.init_conv(x)
        for down_layer in self.downs:
            x = down_layer(x)
        x = self.out_conv(x)  # 输出形状: (batch_size, out_dim, H', W')
        return x


class Unet(nn.Module):
    # 条件生成任务
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            cond_in_dim=1,
            cond_dim=64,
            cond_dim_mults=(2, 4, 8),
            channels=1,
            out_mul=1,
            self_condition=False,
            # self_condition=True,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            window_sizes1=[[16, 16], [8, 8], [4, 4], [2, 2]],
            window_sizes2=[[16, 16], [8, 8], [4, 4], [2, 2]],
            fourier_scale=16,
            ckpt_path=None,
            ignore_keys=[],
            cfg={},
            # debug_mode=False,
            **kwargs
    ):
        super().__init__()

        # determine dimensions
        # self.debug_mode = debug_mode  # 添加调试模式开关
        self.cond_pe = cfg.get('cond_pe', False)
        num_pos_feats = cfg.num_pos_feats if self.cond_pe else 0
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        # self.init_conv_mask = nn.Sequential(
        #     nn.Conv2d(cond_in_dim, cond_dim, 3, padding=1),
        #     nn.GroupNorm(num_groups=min(init_dim // 4, 8), num_channels=init_dim),
        #     nn.SiLU(),
        #     nn.Conv2d(cond_dim, cond_dim, 3, padding=1),
        # )
        # self.init_conv_mask = ConditionEncoder(down_dim_mults=cond_dim_mults, dim=cond_dim,
        #                                        in_dim=cond_in_dim, out_dim=init_dim)

        # 颜色拮抗模块
        self.color_opponency = ColorOpponency()
        # self.color_adjust = nn.Conv2d(5, 3, kernel_size=1)
        self.color_adjust = nn.Conv2d(3, 3, kernel_size=1)
        # self.texture_suppressor = TextureSuppressionLayer(nThetas=8)

        if cfg.cond_net == 'effnet':
            f_condnet = 48
            if cfg.get('without_pretrain', False):
                self.init_conv_mask = efficientnet_b7()
            else:
                self.init_conv_mask = efficientnet_b7(weights=EfficientNet_B7_Weights)
        elif cfg.cond_net == 'resnet':
            f_condnet = 256
            if cfg.get('without_pretrain', False):
                self.init_conv_mask = resnet101()
            else:
                self.init_conv_mask = resnet101(weights=ResNet101_Weights)
        elif cfg.cond_net == 'swin':
            f_condnet = 128
            if cfg.get('without_pretrain', False):
                self.init_conv_mask = swin_b()
            else:
                self.init_conv_mask = swin_b(weights=Swin_B_Weights)
        elif cfg.cond_net == 'vgg':
            f_condnet = 128
            if cfg.get('without_pretrain', False):
                self.init_conv_mask = vgg16()
            else:
                self.init_conv_mask = vgg16(weights=VGG16_Weights)
        else:
            raise NotImplementedError
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels + f_condnet, init_dim, 7, padding=3),
            nn.GroupNorm(num_groups=min(init_dim // 4, 8), num_channels=init_dim),
        )

        if self.cond_pe:
            self.cond_pos_embedding = nn.Sequential(
                PositionEmbeddingLearned(
                    feature_size=cfg.cond_feature_size, num_pos_feats=cfg.num_pos_feats // 2),
                nn.Conv2d(num_pos_feats + init_dim, init_dim, 1)
            )
        # self.init_conv_mask = nn.Conv2d(1, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        dims_rev = dims[::-1]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.projects = nn.ModuleList()
        print(cfg.cond_net)
        if cfg.cond_net == 'effnet':
            self.projects.append(nn.Conv2d(48, dims[0], 1))
            self.projects.append(nn.Conv2d(80, dims[1], 1))
            self.projects.append(nn.Conv2d(224, dims[2], 1))
            self.projects.append(nn.Conv2d(640, dims[3], 1))
            print(len(self.projects))
        elif cfg.cond_net == 'vgg':
            self.projects.append(nn.Conv2d(128, dims[0], 1))
            self.projects.append(nn.Conv2d(256, dims[1], 1))
            self.projects.append(nn.Conv2d(512, dims[2], 1))
            self.projects.append(nn.Conv2d(512, dims[3], 1))
        else:
            self.projects.append(nn.Conv2d(f_condnet, dims[0], 1))
            self.projects.append(nn.Conv2d(f_condnet * 2, dims[1], 1))
            self.projects.append(nn.Conv2d(f_condnet * 4, dims[2], 1))
            self.projects.append(nn.Conv2d(f_condnet * 8, dims[3], 1))
        # print(len(self.projects))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = GaussianFourierProjection(dim // 2, scale=fourier_scale)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.downs_mask = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.relation_layers_down = nn.ModuleList([])
        self.relation_layers_up = nn.ModuleList([])
        self.ups2 = nn.ModuleList([])
        self.relation_layers_up2 = nn.ModuleList([])
        num_resolutions = len(in_out)
        input_size = cfg.get('input_size', [80, 80])
        feature_size_list = [[int(input_size[0] / 2 ** k), int(input_size[1] / 2 ** k)] for k in range(len(dim_mults))]

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))
            # self.downs_mask.append(nn.ModuleList([
            #     block_klass(dim_in, dim_in, time_emb_dim=time_dim),
            #     # block_klass(dim_in, dim_in, time_emb_dim=time_dim),
            #     Residual(PreNorm(dim_in, LinearAttention(dim_in))),
            #     Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            # ]))
            self.relation_layers_down.append(RelationNet(in_channel1=dims[ind], in_channel2=dims[ind], nhead=8,
                                                         layers=1, embed_dim=dims[ind], ffn_dim=dims[ind] * 2,
                                                         window_size1=window_sizes1[ind],
                                                         window_size2=window_sizes2[ind])
                                             )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.decouple1 = nn.Sequential(
            nn.GroupNorm(num_groups=min(mid_dim // 4, 8), num_channels=mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            #BlockFFT(mid_dim, input_size[0]//8, input_size[1]//8),
            BlockGabor(dim=mid_dim, h=input_size[0] // 8, w=input_size[1] // 8, kernel_size=15, sigma=5.0, lambd=10.0,
                       gamma=0.5),
        )
        self.decouple2 = nn.Sequential(
            nn.GroupNorm(num_groups=min(mid_dim // 4, 8), num_channels=mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            #BlockFFT(mid_dim, input_size[0]//8, input_size[1]//8),
            BlockGabor(dim=mid_dim, h=input_size[0] // 8, w=input_size[1] // 8, kernel_size=15, sigma=5.0, lambd=10.0,
                       gamma=0.5),
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))
            self.relation_layers_up.append(RelationNet(in_channel1=dims_rev[ind + 1], in_channel2=dims_rev[ind],
                                                       nhead=8, layers=1, embed_dim=dims_rev[ind],
                                                       ffn_dim=dims_rev[ind] * 2,
                                                       window_size1=window_sizes1[::-1][ind],
                                                       window_size2=window_sizes2[::-1][ind])
                                           )
            self.ups2.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))
            self.relation_layers_up2.append(RelationNet(in_channel1=dims_rev[ind + 1], in_channel2=dims_rev[ind],
                                                        nhead=8, layers=1, embed_dim=dims_rev[ind],
                                                        ffn_dim=dims_rev[ind] * 2,
                                                        window_size1=window_sizes1[::-1][ind],
                                                        window_size2=window_sizes2[::-1][ind])
                                            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim * out_mul, 1)

        self.final_res_block2 = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv2 = nn.Conv2d(dim, self.out_dim, 1)

        # self.init_weights()
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        fix_bb = cfg.get('fix_bb', True)
        if fix_bb:
            for n, p in self.init_conv_mask.named_parameters():
                p.requires_grad = False

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["model"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        msg = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        print('==>Load Unet Info: ', msg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x, time, mask, x_self_cond=None, **kwargs):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
            self.show_tensor_image(x)
        sigma = time.reshape(-1, 1, 1, 1)
        eps = 1e-4
        c_skip1 = 1 - sigma
        c_skip2 = torch.sqrt(sigma)
        # c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_out1 = sigma / torch.sqrt(sigma ** 2 + 1)
        c_out2 = torch.sqrt(1 - sigma) / torch.sqrt(sigma ** 2 + 1)
        c_in = 1

        x_clone = x.clone()
        x = c_in * x

        # mask = torch.cat([], dim=1)
        # 颜色拮抗处理
        # print(f"[mask] shape: {mask.shape}")
        # print(f"[mask] resolution: {mask.shape[2]}x{mask.shape[3]} (HxW)")
        # self.show_tensor_image(mask)
        mask_color = self.color_opponency(mask)  # 打印颜色拮抗后的形状 [1,3,320,320]
        # print(f"color_opponency 输出形状: {mask_color.shape}")  # 应该是 [1,3,320,320]
        mask_color = self.color_adjust(mask_color)  # 打印通道调整后的形状 [1,1,320,320]
        # print(f"color_adjust 输出形状: {mask_color.shape}")  # [1,1,320,320]
        # self.show_tensor_image(mask_color)
        hm = self.init_conv_mask(mask_color)
        #self.show_tensor_image(hm)
        #hm = self.init_conv_mask(mask)
        # 纹理抑制
        # hm = self.texture_suppressor(hm)
        # if self.cond_pe:
        #     m = self.cond_pos_embedding(m)
        # X[1,3,80,80]-->[1,128,80,80]
        x = self.init_conv(torch.cat([x, F.interpolate(hm[0], size=x.shape[-2:], mode="bilinear")], dim=1))

        r = x.clone()

        t = self.time_mlp(torch.log(time) / 4)

        h = []
        h2 = []
        for i, layer in enumerate(self.projects):
            # print(hm[i].shape)
            hm[i] = layer(hm[i])
        hm2 = []
        for i in range(len(hm)):
            hm2.append(hm[i].clone())
        # hm = []
        # hm2 = []
        # [1, 128, 40, 40]-》[1, 256, 20, 20]-》[1, 512, 10, 10]-》[1, 512, 10, 10]
        for i, ((block1, block2, attn, downsample), relation_layer) \
                in enumerate(zip(self.downs, self.relation_layers_down)):
            x = block1(x, t)
            # save_feature_map(x, f'down_{i}/block1')
            h.append(x)
            h2.append(x.clone())
            # m = m_block(m, t)
            # hm.append(m)
            # hm2.append(m.clone())
            x = relation_layer(hm[i], x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            h2.append(x.clone())

            x = downsample(x)
            # save_feature_map(x, f'down_{i}/downsample')
            # m = m_downsample(m)

        # x = x + F.interpolate(hm[-1], size=x.shape[2:], mode="bilinear", align_corners=True)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x1 = x + self.decouple1(x)
        x2 = x + self.decouple2(x)

        x = x1
        for (block1, block2, attn, upsample), relation_layer in zip(self.ups, self.relation_layers_up):
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = relation_layer(hm.pop(), x)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x1 = torch.cat((x, r), dim=1)
        x1 = self.final_res_block(x1, t)
        x1 = self.final_conv(x1)

        x = x2
        for (block1, block2, attn, upsample), relation_layer in zip(self.ups2, self.relation_layers_up2):
            x = torch.cat((x, h2.pop()), dim=1)
            x = block1(x, t)
            x = relation_layer(hm2.pop(), x)
            x = torch.cat((x, h2.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x2 = torch.cat((x, r), dim=1)
        x2 = self.final_res_block2(x2, t)
        x2 = self.final_conv2(x2)
        # sigma = time.reshape(x1.shape[0], *((1,) * (len(x1.shape) - 1)))
        # scale_C = torch.exp(sigma)
        x1 = c_skip1 * x_clone + c_out1 * x1
        x2 = c_skip2 * x_clone + c_out2 * x2
        #print(f"[Debug] x1 shape: {x1.shape}, numel: {x1.numel()}")
        #print(f"[Debug] x2 shape: {x2.shape}, numel: {x2.numel()}")

        #self.show_tensor_image(x1)
        #self.show_tensor_image(x2)
        # 最终输出
        # save_feature_map(x1, 'output/x1')
        # save_feature_map(x2, 'output/x2')
        # return x1, x2, mask_color
        return x1, x2

    def show_tensor_image(self, tensor):
        # 处理 list 包装的情况
        if isinstance(tensor, list):
            if len(tensor) == 0:
                raise ValueError("[show_tensor_image] Received an empty list.")
            tensor = tensor[0]  # 提取第一个 tensor

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"[show_tensor_image] Expected torch.Tensor but got {type(tensor)}")

        img = tensor.detach().cpu().clone()

        if img.ndim == 4:
            img = img[0]  # [C, H, W]
        elif img.ndim == 3:
            pass
        elif img.ndim == 2:
            img = img.unsqueeze(0)  # [1, H, W]
        else:
            raise ValueError(f"[show_tensor_image] Unexpected tensor shape: {img.shape}")

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)  # 转成 RGB
        elif img.shape[0] != 3:
            raise ValueError(f"[show_tensor_image] Unexpected channel size: {img.shape[0]}")

        # 自动创建保存目录
        save_dir = "./debug_images"
        os.makedirs(save_dir, exist_ok=True)

        # 自动编号避免覆盖
        existing = glob.glob(os.path.join(save_dir, "debug_*.png"))
        index = len(existing)
        save_path = os.path.join(save_dir, f"debug_{index:03d}.png")

        # 使用 torchvision 保存图像
        save_image(img, save_path)


if __name__ == "__main__":
    # resnet = resnet101(weights=ResNet101_Weights)
    # effnet = efficientnet_b7(weights=EfficientNet_B7_Weights)
    # effnet = efficientnet_b7(weights=None)
    # x = torch.rand(1, 3, 320, 320)
    # y = effnet(x)
    model = Unet(dim=128, dim_mults=(1, 2, 4, 4),
                 cond_dim=128,
                 cond_dim_mults=(2, 4,),
                 channels=1,
                 window_sizes1=[[8, 8], [4, 4], [2, 2], [1, 1]],
                 window_sizes2=[[8, 8], [4, 4], [2, 2], [1, 1]],
                 cfg=fvcore.common.config.CfgNode({'cond_pe': False, 'input_size': [80, 80],
                                                   'cond_feature_size': (32, 128), 'cond_net': 'swin',
                                                   'num_pos_feats': 96}),
                 # debu g_mode=True
                 )
    x = torch.rand(1, 1, 80, 80)
    mask = torch.rand(1, 3, 320, 320)
    time = torch.tensor([0.5124])
    with torch.no_grad():
        y = model(x, time, mask)
    pass