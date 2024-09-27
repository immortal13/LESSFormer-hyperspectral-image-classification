import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import (CalcAssoc, CalcPixelFeats, CalcSpixelFeats, InitSpixelFeats, RelToAbsIndex, Smear)
from utils import init_grid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def bn_conv_lrelu(in_c, out_c):
    return nn.Sequential(
        nn.BatchNorm2d(in_c), # InstanceNorm2d
        nn.Conv2d(in_c, out_c, 1, padding=0, bias=False),
        nn.ReLU()#nn.LeakyReLU()
    )

def bn_gcconv_lrelu(in_c, out_c, kernel_size, groups):
    return nn.Sequential(
        nn.BatchNorm2d(in_c),
        ## 分组卷积
        nn.Conv2d(in_c, out_c, kernel_size, padding=kernel_size//2, bias=False, groups=groups),
        nn.ReLU()
    )

class FeatureExtactor(nn.Module):
    def __init__(self, n_filters=128, in_ch=202):
        super().__init__()
        self.channel = in_ch
        self.mid_channel = n_filters
        # multiscale CNN
        self.stem = bn_conv_lrelu(self.channel,self.mid_channel)
        self.scale_1 = bn_conv_lrelu(self.mid_channel,self.mid_channel)

    def forward(self, x):
        s = self.stem(x)#(B,C,H,W)       
        s1 = self.scale_1(s)
        return s1


class HSI2Token(nn.Module):
    def __init__(
        self, 
        feat_cvrter,
        n_iters=10, 
        n_spixels=145*145/25,
        n_filters=128, in_ch=202,
        cnn=True, groups=8
    ):
        super().__init__()

        self.n_spixels = n_spixels
        self.n_iters = n_iters

        self.feat_cvrter = feat_cvrter
        self.cnn = cnn
        if cnn:
            # The pixel-wise feature extractor
            self.cnn_modules = FeatureExtactor(n_filters, in_ch)
        else:
            self.cnn_modules = None

        self.gcconv = bn_gcconv_lrelu(n_filters, n_filters, 1, groups)
        self._cached = False
        self._ops = {}
        self._layout = (None, 1, 1)
        self.beta = torch.nn.Parameter(torch.tensor([1.0],requires_grad=True))

    def forward(self, x):
        if self.training:
            # Training mode
            # Use cached objects
            ops, (_, nw_spixels, nh_spixels) = self.get_ops_and_layout(x, ofa=True)
        else:
            # Evaluation mode
            # Every time update the objects
            ops, (_, nw_spixels, nh_spixels) = self.get_ops_and_layout(x, ofa=False)

        # Forward
        pf_ = self.cnn_modules(x) if self.cnn else x
        pf_ = self.gcconv(pf_)
        pf = self.feat_cvrter(pf_, nw_spixels, nh_spixels)
        spf = ops['init_spixels'](pf.detach()) # 迭代时，pf不变？  .detach()

        # Iterations
        for itr in range(self.n_iters):
            Q = self.nd2Q(ops['calc_neg_dist'](pf, spf))
            spf = ops['map_p2sp'](pf, Q)
        spf = ops['map_p2sp'](pf_, Q)
        return Q, ops, x, spf, pf_

    # @staticmethod
    def nd2Q(self, neg_dist):
        # neg_dist = neg_dist/(self.beta**2)
        # Use softmax to compute pixel-superpixel relative soft-associations (degree of membership)
        return F.softmax(neg_dist, dim=1)

    def get_ops_and_layout(self, x, ofa=False):
        if ofa and self._cached:
            return self._ops, self._layout
        
        b, _, h, w = x.size()   # Get size of the input

        # Initialize grid
        init_idx_map, n_spixels, nw_spixels, nh_spixels = init_grid(self.n_spixels, w, h)
        init_idx_map = torch.IntTensor(init_idx_map).expand(b, 1, h, w).to(x.device)

        # Contruct operation modules
        init_spixels = InitSpixelFeats(n_spixels, init_idx_map)
        map_p2sp = CalcSpixelFeats(nw_spixels, nh_spixels, init_idx_map)
        map_sp2p = CalcPixelFeats(nw_spixels, nh_spixels, init_idx_map)
        calc_neg_dist = CalcAssoc(nw_spixels, nh_spixels, init_idx_map)
        map_idx = RelToAbsIndex(nw_spixels, nh_spixels, init_idx_map)
        smear = Smear(n_spixels)

        ops = {
            'init_spixels': init_spixels,
            'map_p2sp': map_p2sp,
            'map_sp2p': map_sp2p,
            'calc_neg_dist': calc_neg_dist,
            'map_idx': map_idx,
            'smear': smear
        }

        if ofa:
            self._ops = ops
            self._layout = (init_idx_map, nw_spixels, nh_spixels)
            self._cached = True

        return ops, (init_idx_map, nw_spixels, nh_spixels)