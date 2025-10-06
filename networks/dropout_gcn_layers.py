#!/usr/bin/env python
# encoding: utf-8
# File Name: gcn_layers.py
# Author: Jiezhong Qiu
# Create Time: 2017/12/18 15:11

from typing import Literal, Optional
import torch
import torch.nn.init as init
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from networks.dropout import (
    DropoutBase,
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


class DropoutBatchGraphConvolution(DropoutBase):
    def __init__(
        self,
        in_features,
        out_features,
        activation: Optional[nn.Module] = None,
        dropout_probability: float = 0.05,
        dropout_inplace: bool = False,
        dropout_type: Literal[
            "alpha", "feature_alpha", "standard"
        ] = "standard",
        bias=True,
    ) -> None:
        super().__init__()

        body = BatchGraphConvolution(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

        dropout = {
            "alpha": nn.AlphaDropout,
            "standard": nn.Dropout,
            "feature_alpha": nn.FeatureAlphaDropout
        }[dropout_type]

        super().build(
            body,
            dropout,
            activation=activation,
            dropout_probability=dropout_probability,
            dropout_inplace=dropout_inplace,
        )