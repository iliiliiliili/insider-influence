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
        all_attentions = {}

        for s in range(samples):
            
            (
                adj,
                x,
            ) = data
            emb = normalized_embedding.float()
            x = torch.cat((x, emb), dim=2)
            bs, n = adj.size()[:2]
            for i, gat_layer in enumerate(self.layer_stack):
                x, attention = gat_layer((x, adj))  # bs x n_head x n x f_out

                if i not in all_attentions:
                    all_attentions[i] = []
                
                all_attentions[i].append(attention)

                if i + 1 == self.n_layer:
                    x = x.mean(dim=1)
                else:
                    x = x.transpose(1, 2).contiguous().view(bs, n, -1)
            outputs.append(F.log_softmax(x, dim=-1)[:, -1, :])
        
        result_var, result = torch.var_mean(
            torch.stack(outputs, dim=0), dim=0, unbiased=False
        )

        if return_uncertainty:

            attentions = {}

            for key, values in all_attentions.items():
                att_var, att = torch.var_mean(
                    torch.stack(values, dim=0), dim=0, unbiased=False
                )
                attentions[key] = (att, att_var)

            return result, result_var, attentions
        else:
            return result


def filter_attentions(att, att_var, limit=0.5):
    att_var = att_var / (att + 1e-6)
    att[att_var > limit] = 0

    return att


class UncertaintyAwareVariationalBatchGAT(nn.Module):
    def __init__(
        self,
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
        super(UncertaintyAwareVariationalBatchGAT, self).__init__()

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

        input_xs = []
        output_xs = []
        all_attentions = {}
        attentions = {}

        (
            adj,
            x,
        ) = data
        emb = normalized_embedding.float()
        x = torch.cat((x, emb), dim=2)
        bs, n = adj.size()[:2]

        for s in range(samples):
            output_xs.append(x)

        for i, gat_layer in enumerate(self.layer_stack):
            input_xs = output_xs
            output_xs = []

            for s in range(samples):
                x = input_xs.pop(0)
                h_prime, attention = gat_layer.attention_step((x, adj))
                x = gat_layer.output_step((h_prime, attention))[0]  # bs x n_head x n x f_out
                # x, attention = gat_layer.all_steps((x, adj))

                if i not in all_attentions:
                    all_attentions[i] = []
                
                all_attentions[i].append(attention)
                output_xs.append(x)

            if i + 1 == self.n_layer:
                for q in range(len(output_xs)):
                    output_xs[q] = output_xs[q].mean(dim=1)
            else:
                for q in range(len(output_xs)):
                    output_xs[q] = output_xs[q].transpose(1, 2).contiguous().view(bs, n, -1)

            att_var, att = torch.var_mean(
                torch.stack(all_attentions[i], dim=0), dim=0, unbiased=False
            )

            att = filter_attentions(att, att_var)

            attentions[i] = (att, att_var)

        for q in range(len(output_xs)):
            output_xs[q] = F.log_softmax(output_xs[q], dim=-1)[:, -1, :]
        
        result_var, result = torch.var_mean(
            torch.stack(output_xs, dim=0), dim=0, unbiased=False
        )

        if return_uncertainty:

            return result, result_var, attentions
        else:
            return result
