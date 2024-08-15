#!/usr/bin/env python
# encoding: utf-8
# File Name: gcn_layers.py
# Author: Jiezhong Qiu
# Create Time: 2017/12/18 15:11

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from networks.variational import (
    VariationalBase,
    init_weights as vnn_init_weights,
)


class BatchGraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(BatchGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)
        init.xavier_uniform_(self.weight)

    def forward(self, inputs):
        x, lap = inputs
        expand_weight = self.weight.expand(x.shape[0], -1, -1)
        support = torch.bmm(x, expand_weight)
        output = torch.bmm(lap, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class VariationalBatchGraphConvolution(VariationalBase):
    def __init__(
        self,
        in_features,
        out_features,
        activation,
        bias=True,
        activation_mode="mean",
        global_std_mode="none",
    ) -> None:
        super().__init__()

        means = BatchGraphConvolution(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

        if global_std_mode == "replace":
            stds = None
        else:
            stds = BatchGraphConvolution(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
            )

        super().build(
            means,
            stds,
            None,
            None,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=False,
            batch_norm_mode=None,
            global_std_mode=global_std_mode,
        )