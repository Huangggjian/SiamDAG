# -----------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by HUANG JIAN (jian.huang@hdu.edu.cn)
# ------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
from .DAG import DAG_


from .siamfc import SiamFC_
from .myconnect import box_tower, AdjustLayer, Corr_Up, MultiDiCorr, DAGCorr
from .backbones import ResNet50, ResNet22W
#from .mask import MultiRefine, MultiRefineTRT
from .modules import MultiFeatureBase
from collections import OrderedDict
import os
import sys
sys.path.append('../lib')


# ---------------------------
class DAG(DAG_):
    def __init__(self, align=False, online=False):
        super(DAG, self).__init__()
        self.features = ResNet50(used_layers=[3], online=online)   # in param     
        self.neck = AdjustLayer(in_channels=1024, out_channels=256)
        self.connect_model = box_tower(inchannels=256, outchannels=256, towernum=4)
        self.align_head = None




