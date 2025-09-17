#!/usr/bin/env python
# encoding: utf-8
# File Name: gat_layers.py
# Author: Jiezhong Qiu
# Create Time: 2017/12/18 15:11


from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from networks.variational import (
    VariationalBase,
    MultiOutputVariationalBase,
    init_weights as vnn_init_weights,
    multi_output_variational_forward,
    multi_output_variational_gaussian_sample,
)


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, input):
        (h, adj) = input
        bs, n = h.size()[:2]  # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = 1 - adj.unsqueeze(1)  # bs x 1 x n x n
        attn.data.masked_fill_(mask.bool(), float("-inf"))
        attn = self.softmax(attn)  # bs x n_head x n x n
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        if self.bias is not None:
            return [output + self.bias, attn]
        else:
            return [output, attn]

    def attention_step(self, input):
        (h, adj) = input
        bs, n = h.size()[:2]  # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = 1 - adj.unsqueeze(1)  # bs x 1 x n x n
        attn.data.masked_fill_(mask.bool(), float("-inf"))
        attn = self.softmax(attn)  # bs x n_head x n x n

        return h_prime, attn

    def output_step(self, input):
        (h_prime, attn) = input
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        if self.bias is not None:
            return [output + self.bias]
        else:
            return [output]


class VariationalBatchMultiHeadGraphAttention(MultiOutputVariationalBase):
    def __init__(
        self,
        n_head,
        f_in,
        f_out,
        activation,
        bias=True,
        activation_mode="mean",
        global_std_mode="none",
        batch_norm_mode="mean",
        use_batch_norm=False,
    ) -> None:
        super().__init__()

        self.has_bias = bias

        means = BatchMultiHeadGraphAttention(
            n_head=n_head,
            f_in=f_in,
            f_out=f_out,
            bias=bias,
        )

        if global_std_mode == "replace":
            stds = None
        else:
            stds = BatchMultiHeadGraphAttention(
                n_head=n_head,
                f_in=f_in,
                f_out=f_out,
                bias=bias,
            )

        super().build(
            means,
            stds,
            None,
            None,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            global_std_mode=global_std_mode,
        )

    def means_attention_step(self):
        if isinstance(self.means, nn.Sequential):

            return self.means[0].attention_step

        return self.means.attention_step
    def means_output_step(self):
        if isinstance(self.means, nn.Sequential):

            def run_means_output_step(*args, **kwargs):

                x = self.means[0].output_step(*args, **kwargs)

                if isinstance(x, (tuple, list)):
                    for i in range(1, len(self.means)):
                        x = [self.means[i](elem) for elem in x]
                else:
                    for i in range(1, len(self.means)):
                        x = self.means[i](x)
                
                return x

            return run_means_output_step
        
        return self.means.output_step
    
    def stds_output_step(self):
        if isinstance(self.stds, nn.Sequential):

            def run_stds_output_step(*args, **kwargs):

                x = self.stds[0].output_step(*args, **kwargs)

                if isinstance(x, (tuple, list)):
                    for i in range(1, len(self.stds)):
                        x = [self.stds[i](elem) for elem in x]
                else:
                    for i in range(1, len(self.stds)):
                        x = self.stds[i](x)
                
                return x

            return run_stds_output_step
        
        return self.stds.output_step

    def stds_attention_step(self):
        if isinstance(self.stds, nn.Sequential):
            
            return self.stds[0].attention_step
        
        return self.stds.attention_step if self.stds else None

    def attention_step(self, input):

        return multi_output_variational_forward(
            self.means_attention_step(),
            self.stds_attention_step(),
            input,
            self.global_std_mode,
            self.LOG_STDS,
            VariationalBase.FIX_GAUSSIAN,
            VariationalBase.GLOBAL_STD,
            self.end_batch_norm,
            self.end_activation,
            fix_gaussian_storage = self,
        )

    def raw_attention_step(self, input):
        means = self.means_attention_step()(input)
        stds = self.stds_attention_step()(input)
        return (means, stds)

    def output_step(self, input):
        return self.means_output_step()(input)
        # return multi_output_variational_forward(
        #     self.means.output_step,
        #     self.stds.output_step,
        #     input,
        #     self.global_std_mode,
        #     self.LOG_STDS,
        #     VariationalBase.FIX_GAUSSIAN,
        #     VariationalBase.GLOBAL_STD,
        #     self.end_batch_norm,
        #     self.end_activation,
        # )
    
    def variational_output_step(self, input):

        means = self.means_output_step()(input[0])
        stds = self.stds_output_step()(input[1])

        return multi_output_variational_gaussian_sample(
            means, stds, self.global_std_mode, VariationalBase.GLOBAL_STD, VariationalBase.FIX_GAUSSIAN, 
            fix_gaussian_storage = self,
        )
    
    def variational_attention_sample(self, attention_means, attention_stds):

        return multi_output_variational_gaussian_sample(
            attention_means, attention_stds, self.global_std_mode, VariationalBase.GLOBAL_STD, VariationalBase.FIX_GAUSSIAN, 
            fix_gaussian_storage = self,
        )

    def all_steps(self, input):
        return multi_output_variational_forward(
            self.means,
            self.stds,
            input,
            self.global_std_mode,
            self.LOG_STDS,
            VariationalBase.FIX_GAUSSIAN,
            VariationalBase.GLOBAL_STD,
            self.end_batch_norm,
            self.end_activation,
            fix_gaussian_storage = self,
        )


    def _init_weights(self):

        all_submodules = [
            lambda x: (x.w, True),
            lambda x: (x.a_src, True),
            lambda x: (x.a_dst, True),
        ]

        if self.has_bias:
            all_submodules.append(lambda x: (x.bias, False))

        vnn_init_weights(self, all_submodules)
