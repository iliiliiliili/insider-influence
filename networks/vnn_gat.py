import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.variational import VariationalBase
from networks.vnn_gat_layers import VariationalBatchMultiHeadGraphAttention


class VariationalBatchGAT(nn.Module):
    def __init__(
        self,  # pretrained_emb,
        n_units,
        n_heads,
        fine_tune=False,
        instance_normalization=True,
        FIX_GAUSSIAN=None,
        INIT_WEIGHTS="usual",
        GLOBAL_STD=0,
        activation_mode="mean",
        batch_norm_mode="mean",
        global_std_mode="none",
        use_batch_norm=False,
        samples=4,
        test_samples=4,
    ):
        super(VariationalBatchGAT, self).__init__()

        self.default_samples = samples
        self.test_samples = test_samples

        VariationalBase.FIX_GAUSSIAN = FIX_GAUSSIAN
        VariationalBase.INIT_WEIGHTS = INIT_WEIGHTS
        VariationalBase.GLOBAL_STD = GLOBAL_STD

        if VariationalBase.FIX_GAUSSIAN is not None:
            print("FIX_GAUSSIAN", VariationalBase.FIX_GAUSSIAN)
            print("FIX_GAUSSIAN", VariationalBase.FIX_GAUSSIAN)
            print("FIX_GAUSSIAN", VariationalBase.FIX_GAUSSIAN)
            print("FIX_GAUSSIAN", VariationalBase.FIX_GAUSSIAN)

        self.n_layer = len(n_units) - 1
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(
                64, momentum=0.0, affine=True  # pretrained_emb.size(1),
            )

        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/2
        # For the public data this is not necessary to train, but since our
        # models were trained with non-public data this is left here.
        # 1790 is the total number of nodes in the network.
        self.embedding = nn.Embedding(1790, 64)
        self.embedding.weight.requires_grad = fine_tune
        n_units[0] += 64

        self.layer_stack = nn.ModuleList()
        for i in range(self.n_layer):
            # consider multi head from last layer
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            activation = torch.nn.ELU() if i + 1 < self.n_layer else None
            self.layer_stack.append(
                VariationalBatchMultiHeadGraphAttention(
                    n_heads[i],
                    f_in=f_in,
                    f_out=n_units[i + 1],
                    activation=activation,
                    activation_mode=activation_mode,
                    batch_norm_mode=batch_norm_mode,
                    global_std_mode=global_std_mode,
                    use_batch_norm=use_batch_norm,
                )
            )

    def forward(self, data, normalized_embedding=None, samples=None, return_uncertainty=False):

        if samples is None:
            if self.training:
                samples = self.default_samples
            else:
                samples = self.test_samples

        outputs = []

        for s in range(samples):
            
            (
                adj,
                x,
            ) = data
            emb = normalized_embedding.float()
            x = torch.cat((x, emb), dim=2)
            bs, n = adj.size()[:2]
            for i, gat_layer in enumerate(self.layer_stack):
                x = gat_layer((x, adj))  # bs x n_head x n x f_out
                if i + 1 == self.n_layer:
                    x = x.mean(dim=1)
                else:
                    x = x.transpose(1, 2).contiguous().view(bs, n, -1)
            outputs.append(F.log_softmax(x, dim=-1)[:, -1, :])
        
        result_var, result = torch.var_mean(
            torch.stack(outputs, dim=0), dim=0, unbiased=False
        )

        if return_uncertainty:
            return result, result_var
        else:
            return result
