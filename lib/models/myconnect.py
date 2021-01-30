# -----------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by HUANG JIAN (jian.huang@hdu.edu.cn)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SKConv_2
class Corr_Up(nn.Module):
    """
    SiamFC head
    """

    def __init__(self):
        super(Corr_Up, self).__init__()

    def _conv2d_group(self, x, kernel):
        batch = x.size()[0]
        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
        px = x.view(1, -1, x.size()[2], x.size()[3])
        po = F.conv2d(px, pk, groups=batch)
        po = po.view(batch, -1, po.size()[2], po.size()[3])
        return po

    def forward(self, z_f, x_f):
        if not self.training:
            return 0.1 * F.conv2d(x_f, z_f)
        else:
            return 0.1 * self._conv2d_group(x_f, z_f)


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class MultiDiCorr(nn.Module):
    """
    For tensorRT version
    """

    def __init__(self, inchannels=512, outchannels=256):
        super(MultiDiCorr, self).__init__()
        self.cls_encode = matrix(in_channels=inchannels, out_channels=outchannels)
        self.reg_encode = matrix(in_channels=inchannels, out_channels=outchannels)

    def forward(self, search, kernal):
        """
        :param search:
        :param kernal:
        :return:  for tensor2trt
        """
        cls_z0, cls_z1, cls_z2, cls_x0, cls_x1, cls_x2 = self.cls_encode(kernal, search)  # [z11, z12, z13]
        reg_z0, reg_z1, reg_z2, reg_x0, reg_x1, reg_x2 = self.reg_encode(kernal, search)  # [x11, x12, x13]

        return cls_z0, cls_z1, cls_z2, cls_x0, cls_x1, cls_x2, reg_z0, reg_z1, reg_z2, reg_x0, reg_x1, reg_x2


class DAGCorr(nn.Module):
    """
    For tensorRT version
    """

    def __init__(self, inchannels=512):
        super(DAGCorr, self).__init__()

        self.cls_dw = GroupDW(in_channels=inchannels)
        self.reg_dw = GroupDW(in_channels=inchannels)

    def forward(self, cls_z, cls_x, reg_z, reg_x):
        cls_dw = self.cls_dw(cls_z, cls_x)
        reg_dw = self.reg_dw(reg_z, reg_x)

        return cls_dw, reg_dw


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):  # 1024  256
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, crop=False):
        x_ori = self.downsample(x)
        if x_ori.size(3) < 20 and crop:
            l = 4
            r = -4
            xf = x_ori[:, :, l:r, l:r]

        if not crop:
            return x_ori
        else:
            return x_ori, xf


# --------------------
# DAG module
# --------------------
class matrix(nn.Module):
    """
    encode backbone feature
    """

    def __init__(self, in_channels, out_channels):
        super(matrix, self).__init__()
        assert in_channels == out_channels, 'channels not matched'
        self.matrix11_k = SKConv_2(in_channels)
        self.matrix11_s = SKConv_2(in_channels)
        

    def forward(self, z, x):
        z11 = self.matrix11_k(z)
        x11 = self.matrix11_s(x)
        return [z11], [x11]




class GroupDW(nn.Module):
    """
    encode backbone feature
    """

    def __init__(self, in_channels=256):
        super(GroupDW, self).__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, z, x):
       
        z11 = z[0]
        x11 = x[0]
        re11 = xcorr_depthwise(x11, z11)  # reg_z and reg_x or cls_z and cls_x
        re = [re11]
        # weight
        weight = F.softmax(self.weight, 0)

        s = 0
        for i in range(1):
            s += weight[i] * re[i]

        return s


class SingleDW(nn.Module):
    """
    encode backbone feature
    """

    def __init__(self, in_channels=256):
        super(SingleDW, self).__init__()

    def forward(self, z, x):
        s = xcorr_depthwise(x, z)

        return s


class box_tower(nn.Module):
    """
    box tower for FCOS reg
    """

    def __init__(self, inchannels=256, outchannels=256, towernum=1):  # 256 256 3
        super(box_tower, self).__init__()
        tower = []
        cls_tower = []
       
        # encode backbone
        self.cls_encode = matrix(in_channels=inchannels, out_channels=outchannels)  # [z11, ], [x11,]
        self.reg_encode = matrix(in_channels=inchannels, out_channels=outchannels)  # [z11, ], [x11,]
        self.cls_dw = GroupDW(in_channels=inchannels)
        self.reg_dw = GroupDW(in_channels=inchannels)

     
        # box pred head
        for i in range(towernum):
            if i == 0:
                tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            else:
                tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))

            tower.append(nn.BatchNorm2d(outchannels))
            tower.append(nn.ReLU())

        # cls tower
        for i in range(towernum):
            if i == 0:
                cls_tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))
            else:
                cls_tower.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1))

            cls_tower.append(nn.BatchNorm2d(outchannels))
            cls_tower.append(nn.ReLU())

        self.add_module('bbox_tower', nn.Sequential(*tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))

   
        self.bbox_pred = nn.Conv2d(outchannels, 4, kernel_size=3, stride=1, padding=1)

        self.cls_pred = nn.Conv2d(outchannels, 1, kernel_size=3, stride=1, padding=1)

        self.adjust = nn.Parameter(0.1 * torch.ones(1))
        self.bias = nn.Parameter(torch.Tensor(1.0 * torch.ones(1, 4, 1, 1)).cuda())
        self.quality = nn.Conv2d(outchannels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, search, kernal, update=None):

      
        if update is None:
            cls_z, cls_x = self.cls_encode(kernal, search) 

        else:
            cls_z, cls_x = self.cls_encode(update, search)

        reg_z, reg_x = self.reg_encode(kernal, search) 

        # cls and reg DW
        cls_dw = self.cls_dw(cls_z, cls_x)
        reg_dw = self.reg_dw(reg_z, reg_x)

        # --------------reg-------------------
        x_reg = self.bbox_tower(reg_dw)  
        x = self.adjust * self.bbox_pred(x_reg) + self.bias  
        x = torch.exp(x)

        # --------------cls------------------
        c = self.cls_tower(cls_dw)    
        cls = 0.1 * self.cls_pred(c)  

        quailty_branch = 0.1 * self.quality(x_reg)
        
        return x, cls, cls_dw, x_reg, quailty_branch

