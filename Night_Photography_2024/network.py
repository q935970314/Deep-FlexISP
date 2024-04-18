import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Module
import functools
from torch.optim import lr_scheduler
from collections import OrderedDict
import numpy as np

'''
# ===================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules
# to a single nn.Sequential
# ===================================
'''

def seq(*args):
    if len(args) == 1:
        args = args[0]
    if isinstance(args, nn.Module):
        return args
    modules = OrderedDict()
    if isinstance(args, OrderedDict):
        for k, v in args.items():
            modules[k] = seq(v)
        return nn.Sequential(modules)
    assert isinstance(args, (list, tuple))
    return nn.Sequential(*[seq(i) for i in args])

'''
# ===================================
# Useful blocks
# --------------------------------
# conv (+ normaliation + relu)
# concat
# sum
# resblock (ResBlock)
# resdenseblock (ResidualDenseBlock_5C)
# resinresdenseblock (RRDB)
# ===================================
'''

# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
         output_padding=0, dilation=1, groups=1, bias=True,
         padding_mode='zeros', mode='CBR'):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=bias,
                               padding_mode=padding_mode))
        elif t == 'X':
            assert in_channels == out_channels
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=in_channels,
                               bias=bias,
                               padding_mode=padding_mode))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        output_padding=output_padding,
                                        groups=groups,
                                        bias=bias,
                                        dilation=dilation,
                                        padding_mode=padding_mode))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'i':
            L.append(nn.InstanceNorm2d(out_channels))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'P':
            L.append(nn.PReLU())
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return seq(*L)

# -------------------------------------------------------
# Concat the output of a submodule to its input
# -------------------------------------------------------
class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        return self.sub.__repr__() + '_concat'

# -------------------------------------------------------
# Elementwise sum the output of a submodule to its input
# -------------------------------------------------------
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

class DWTForward(nn.Module):
    def __init__(self):
        super(DWTForward, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                            hl[None,::-1,::-1], hh[None,::-1,::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)
    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y

class DWTInverse(nn.Module):
    def __init__(self):
        super(DWTInverse, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                            hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = int(x.shape[1] / 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        return y

# -------------------------------------------------------
# Channel Attention (CA) Layer
# -------------------------------------------------------
class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

# -------------------------------------------------------
# Content Unrelated Channel Attention (CUCA) Layer
# -------------------------------------------------------
class CUCALayer(nn.Module):
    def __init__(self, channel=64, min=0, max=None):
        super(CUCALayer, self).__init__()

        self.attention = nn.Conv2d(channel, channel, 1, padding=0,
                                   groups=channel, bias=False)
        self.min, self.max = min, max
        nn.init.uniform_(self.attention.weight, 0, 1)

    def forward(self, x):
        self.attention.weight.data.clamp_(self.min, self.max)
        return self.attention(x)


# -------------------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC'):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size,
                        stride, padding, bias=bias, mode=mode)

    def forward(self, x):
        res = self.res(x)
        return x + res

# -------------------------------------------------------
# Residual Channel Attention Block (RCAB)
# -------------------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC', reduction=16):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size,
                        stride, padding, bias=bias, mode=mode)
        # self.CA = CALayer(out_channels, reduction)
        #self.SA = spatial_attn_layer()            ## Spatial Attention
        #self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

    def forward(self, x):
        res = self.res(x)
        #sa_branch = self.SA(res)
        # ca_branch = self.CA(res)
        #res = torch.cat([sa_branch, ca_branch], dim=1)
        #res = self.conv1x1(res)
        return res + x


# -------------------------------------------------------
# Residual Channel Attention Group (RG)
# -------------------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC', reduction=16, nb=12, num_attention_block=4, use_attention=True):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        RG = []
        for _ in range(num_attention_block):
            RG.extend([RCABlock(in_channels, out_channels, kernel_size, stride, padding,
                        bias, mode, reduction) for _ in range(nb//num_attention_block)])
            if use_attention:
                RG.append(AttentionResBlock(in_channels))
        RG.append(conv(out_channels, out_channels, mode='C'))

        # self.rg = ShortcutBlock(nn.Sequential(*RG))
        self.rg = nn.Sequential(*RG)


    def forward(self, x):
        res = self.rg(x)
        return res + x
    
# -------------------------------------------------------
# conv + subp + relu
# -------------------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3,
                          stride=1, padding=1, bias=True, mode='2R'):
    # mode examples: 2, 2R, 2BR, 3, ..., 4BR.
    assert len(mode)<4 and mode[0] in ['2', '3', '4']
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size,
               stride, padding, bias=bias, mode='C'+mode)
    return up1


# -------------------------------------------------------
# nearest_upsample + conv + relu
# -------------------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1,
                    padding=1, bias=True, mode='2R'):
    # mode examples: 2, 2R, 2BR, 3, ..., 3BR.
    assert len(mode)<4 and mode[0] in ['2', '3']
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride,
               padding, bias=bias, mode=mode)
    return up1


# -------------------------------------------------------
# convTranspose + relu
# -------------------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2,
                           stride=2, padding=0, bias=True, mode='2R'):
    # mode examples: 2, 2R, 2BR, 3, ..., 4BR.
    assert len(mode)<4 and mode[0] in ['2', '3', '4']
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride,
               padding, bias=bias, mode=mode)
    return up1


'''
# ======================
# Downsampler
# ======================
'''


# -------------------------------------------------------
# strideconv + relu
# -------------------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2,
                          stride=2, padding=0, bias=True, mode='2R'):
    # mode examples: 2, 2R, 2BR, 3, ..., 4BR.
    assert len(mode)<4 and mode[0] in ['2', '3', '4']
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride,
                 padding, bias=bias, mode=mode)
    return down1


# -------------------------------------------------------
# maxpooling + conv + relu
# -------------------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3,
                       stride=1, padding=0, bias=True, mode='2R'):
    # mode examples: 2, 2R, 2BR, 3, ..., 3BR.
    assert len(mode)<4 and mode[0] in ['2', '3']
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride,
                     padding, bias=bias, mode=mode[1:])
    return sequential(pool, pool_tail)


# -------------------------------------------------------
# averagepooling + conv + relu
# -------------------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3,
                       stride=1, padding=1, bias=True, mode='2R'):
    # mode examples: 2, 2R, 2BR, 3, ..., 3BR.
    assert len(mode)<4 and mode[0] in ['2', '3']
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride,
                     padding, bias=bias, mode=mode[1:])
    return sequential(pool, pool_tail)



class AttentionResBlock(nn.Module):
    def __init__(self, dim: int):
        super(AttentionResBlock, self).__init__()
        self._spatial_attention_conv = nn.Conv2d(2, dim, kernel_size=3, padding=1)

        # Channel attention MLP
        self._channel_attention_conv0 = nn.Conv2d(1, dim, kernel_size=1, padding=0)
        self._channel_attention_conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)

        self._out_conv = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        # Spatial attention
        mean = torch.mean(x, dim=1, keepdim=True)  # Mean/Max on C axis
        max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.cat([mean, max], dim=1)  # [B, 2, H, W]
        spatial_attention = self._spatial_attention_conv(spatial_attention)
        spatial_attention = torch.sigmoid(spatial_attention) * x

        channel_attention = torch.relu(self._channel_attention_conv0(mean))
        channel_attention = self._channel_attention_conv1(channel_attention)
        channel_attention = torch.sigmoid(channel_attention) * x

        attention = torch.cat([spatial_attention, channel_attention], dim=1)  # [B, 2*dim, H, W]
        attention = self._out_conv(attention)
        return x + attention


class MWRCANv2(nn.Module):
    def __init__(self):
        super(MWRCANv2, self).__init__()
        c1 = 64
        c2 = 96
        c3 = 128
        n_b = 16

        self.head = seq(
            nn.PixelUnshuffle(2),
            DWTForward()
        )

        self.down1 = seq(
            nn.Conv2d(48, c1, 3, 1, 1),
            nn.PReLU(),
            RCAGroup(in_channels=c1, out_channels=c1, nb=n_b, num_attention_block=4)
        )

        self.down2 = seq(
            DWTForward(),
            nn.Conv2d(c1 * 4, c2, 3, 1, 1),
            nn.PReLU(),
            RCAGroup(in_channels=c2, out_channels=c2, nb=n_b, num_attention_block=4)
        )

        self.down3 = seq(
            DWTForward(),
            nn.Conv2d(c2 * 4, c3, 3, 1, 1),
            nn.PReLU()
        )

        self.middle = seq(
            RCAGroup(in_channels=c3, out_channels=c3, nb=n_b, num_attention_block=4),
            RCAGroup(in_channels=c3, out_channels=c3, nb=n_b, num_attention_block=4)
        )

        self.up1 = seq(
            nn.Conv2d(c3, c2 * 4, 3, 1, 1),
            nn.PReLU(),
            DWTInverse()
        )

        self.up2 = seq(
            RCAGroup(in_channels=c2, out_channels=c2, nb=n_b, num_attention_block=4),
            nn.Conv2d(c2, c1 * 4, 3, 1, 1),
            nn.PReLU(),
            DWTInverse()
        )

        self.up3 = seq(
            RCAGroup(in_channels=c1, out_channels=c1, nb=n_b, num_attention_block=4),
            nn.Conv2d(c1, 48, 3, 1, 1)
        )

        self.tail = seq(
            DWTInverse(),
            nn.PixelShuffle(2)
        )

    def forward(self, x, c=None):
        c1 = self.head(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        m = self.middle(c4)
        c5 = self.up1(m) + c3
        c6 = self.up2(c5) + c2
        c7 = self.up3(c6) + c1
        out = self.tail(c7)

        return out



class MWRCANv3(nn.Module):
    def __init__(self):
        super(MWRCANv3, self).__init__()
        c1 = 64
        c2 = 96
        c3 = 128
        n_b = 16

        self.head = seq(
            DWTForward()
        )

        self.down1 = seq(
            nn.Conv2d(12, c1, 3, 1, 1),
            nn.PReLU(),
            RCAGroup(in_channels=c1, out_channels=c1, nb=n_b)
        )

        self.down2 = seq(
            DWTForward(),
            nn.Conv2d(c1 * 4, c2, 3, 1, 1),
            nn.PReLU(),
            RCAGroup(in_channels=c2, out_channels=c2, nb=n_b)
        )

        self.down3 = seq(
            DWTForward(),
            nn.Conv2d(c2 * 4, c3, 3, 1, 1),
            nn.PReLU()
        )

        self.middle = seq(
            RCAGroup(in_channels=c3, out_channels=c3, nb=n_b),
            RCAGroup(in_channels=c3, out_channels=c3, nb=n_b)
        )

        self.up1 = seq(
            nn.Conv2d(c3, c2 * 4, 3, 1, 1),
            nn.PReLU(),
            DWTInverse()
        )

        self.up2 = seq(
            RCAGroup(in_channels=c2, out_channels=c2, nb=n_b),
            nn.Conv2d(c2, c1 * 4, 3, 1, 1),
            nn.PReLU(),
            DWTInverse()
        )

        self.up3 = seq(
            RCAGroup(in_channels=c1, out_channels=c1, nb=n_b),
            nn.Conv2d(c1, 12, 3, 1, 1)
        )

        self.tail = seq(
            DWTInverse()
        )

    def forward(self, x, c=None):
        c1 = self.head(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        m = self.middle(c4)
        c5 = self.up1(m) + c3
        c6 = self.up2(c5) + c2
        c7 = self.up3(c6) + c1
        out = self.tail(c7)

        return out


class MWRCANv4(nn.Module):
    def __init__(self, c1 = 64, c2 = 96, c3 = 128, n_b = 16):
        super(MWRCANv4, self).__init__()

        self.head = seq(
            DWTForward()
        )

        self.down1 = seq(
            nn.Conv2d(12, c1, 3, 1, 1),
            nn.PReLU(),
            RCAGroup(in_channels=c1, out_channels=c1, nb=n_b, use_attention=False)
        )

        self.down2 = seq(
            DWTForward(),
            nn.Conv2d(c1 * 4, c2, 3, 1, 1),
            nn.PReLU(),
            RCAGroup(in_channels=c2, out_channels=c2, nb=n_b, use_attention=False)
        )

        self.down3 = seq(
            DWTForward(),
            nn.Conv2d(c2 * 4, c3, 3, 1, 1),
            nn.PReLU()
        )

        self.middle = seq(
            RCAGroup(in_channels=c3, out_channels=c3, nb=n_b, use_attention=False),
            RCAGroup(in_channels=c3, out_channels=c3, nb=n_b, use_attention=False)
        )

        self.up1 = seq(
            nn.Conv2d(c3, c2 * 4, 3, 1, 1),
            nn.PReLU(),
            DWTInverse()
        )

        self.up2 = seq(
            RCAGroup(in_channels=c2, out_channels=c2, nb=n_b, use_attention=False),
            nn.Conv2d(c2, c1 * 4, 3, 1, 1),
            nn.PReLU(),
            DWTInverse()
        )

        self.up3 = seq(
            RCAGroup(in_channels=c1, out_channels=c1, nb=n_b, use_attention=False),
            nn.Conv2d(c1, 12, 3, 1, 1)
        )

        self.tail = seq(
            DWTInverse()
        )

    def forward(self, x, c=None):
        c1 = self.head(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        m = self.middle(c4)
        c5 = self.up1(m) + c3
        c6 = self.up2(c5) + c2
        c7 = self.up3(c6)
        out = self.tail(c7)

        return out