from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.dropout import DropoutBase
from networks.dropout_gat_layers import DropoutBatchMultiHeadGraphAttention


class DropoutBatchGAT(nn.Module):
    def __init__(
        self,  # pretrained_emb,
        n_units,
        n_heads,
        fine_tune=False,
        instance_normalization=True,
        dropout_probability: float = 0.05,
        dropout_inplace: bool = False,
        dropout_type: Literal[
            "alpha", "feature_alpha", "standard"
        ] = "standard",
        samples=4,
        test_samples=4,
    ):
        super(DropoutBatchGAT, self).__init__()

        self.default_samples = samples
        self.test_samples = test_samples

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
        # n_units[0] += 64

        self.layer_stack = nn.ModuleList()
        for i in range(self.n_layer):
            # consider multi head from last layer
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            activation = torch.nn.ELU() if i + 1 < self.n_layer else None
            self.layer_stack.append(
                DropoutBatchMultiHeadGraphAttention(
                    n_heads[i],
                    f_in=f_in,
                    f_out=n_units[i + 1],
                    activation=activation,
                    dropout_probability=dropout_probability,
                    dropout_inplace=dropout_inplace,
                    dropout_type=dropout_type,
                )
            )

    def forward(
        self, data, normalized_embedding=None, samples=None, return_uncertainty=False
    ):

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


def filter_attentions(att, att_var, limit=0.5, filtered_value=0.01):
    att_var = att_var / (att + 1e-6)

    if limit > 0:
        att[att_var > limit] = filtered_value
        att_var[att_var > limit] = 0
    else:
        att[att_var < -limit] = filtered_value
        att_var[att_var < -limit] = 0

    return att, att_var


class UncertaintyAwareEarlyAttentionVariationalBatchGAT(nn.Module):
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
        training_method="variational",
        attention_filter_limit=0.5,
        variational_mode_on_inference=False,
    ):
        super(UncertaintyAwareEarlyAttentionVariationalBatchGAT, self).__init__()

        self.default_samples = samples
        self.test_samples = test_samples
        self.training_method = training_method
        self.attention_filter_limit = attention_filter_limit
        self.variational_mode_on_inference = variational_mode_on_inference

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
        # n_units[0] += 64

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

    def forward_variational(
        self, data, normalized_embedding=None, samples=None, return_uncertainty=False
    ):

        if samples is None:
            if self.training:
                samples = self.default_samples
            else:
                samples = self.test_samples

        input_xs = []
        output_xs = []
        h_primes = []
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
            h_primes = []

            for s in range(samples):
                x = input_xs.pop(0)
                h_prime, attention = gat_layer.attention_step((x, adj))
                x = gat_layer.output_step((h_prime, attention))[
                    0
                ]  # bs x n_head x n x f_out
                output_xs.append(x)

            if i + 1 == self.n_layer:
                for q in range(len(output_xs)):
                    output_xs[q] = output_xs[q].mean(dim=1)
            else:
                for q in range(len(output_xs)):
                    output_xs[q] = (
                        output_xs[q].transpose(1, 2).contiguous().view(bs, n, -1)
                    )

        for q in range(len(output_xs)):
            output_xs[q] = F.log_softmax(output_xs[q], dim=-1)[:, -1, :]

        result_var, result = torch.var_mean(
            torch.stack(output_xs, dim=0), dim=0, unbiased=False
        )

        if return_uncertainty:

            return result, result_var, attentions
        else:
            return result

    def forward_uncertainty_aware(
        self, data, normalized_embedding=None, samples=None, return_uncertainty=False
    ):

        if samples is None:
            if self.training:
                samples = self.default_samples
            else:
                samples = self.test_samples

        input_xs = []
        output_xs = []
        h_primes = []
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
            h_primes = []

            for s in range(samples):
                x = input_xs.pop(0)
                h_prime, attention = gat_layer.attention_step((x, adj))
                h_primes.append(h_prime)
                # x = gat_layer.output_step((h_prime, attention))[0]  # bs x n_head x n x f_out
                # x, attention = gat_layer.all_steps((x, adj))

                if i not in all_attentions:
                    all_attentions[i] = []

                all_attentions[i].append(attention)
                # output_xs.append(x)

            att_var, att = torch.var_mean(
                torch.stack(all_attentions[i], dim=0), dim=0, unbiased=False
            )

            attention_filtered, _ = filter_attentions(att, att_var, self.attention_filter_limit)

            attentions[i] = (att, att_var)

            for s in range(samples):
                h_prime = h_primes.pop(0)
                x = gat_layer.output_step((h_prime, attention_filtered))[
                    0
                ]  # bs x n_head x n x f_out
                output_xs.append(x)

            if i + 1 == self.n_layer:
                for q in range(len(output_xs)):
                    output_xs[q] = output_xs[q].mean(dim=1)
            else:
                for q in range(len(output_xs)):
                    output_xs[q] = (
                        output_xs[q].transpose(1, 2).contiguous().view(bs, n, -1)
                    )

        for q in range(len(output_xs)):
            output_xs[q] = F.log_softmax(output_xs[q], dim=-1)[:, -1, :]

        result_var, result = torch.var_mean(
            torch.stack(output_xs, dim=0), dim=0, unbiased=False
        )

        if return_uncertainty:

            return result, result_var, attentions
        else:
            return result

    def forward(
        self, data, normalized_embedding=None, samples=None, return_uncertainty=False
    ):

        if self.training:
            if self.training_method == "variational":
                return self.forward_variational(
                    data, normalized_embedding, samples, return_uncertainty
                )
            elif self.training_method == "uncertainty_aware":
                return self.forward_uncertainty_aware(
                    data, normalized_embedding, samples, return_uncertainty
                )
            else:
                raise ValueError("Invalid training method")
        else:

            if self.variational_mode_on_inference:
                return self.forward_variational(
                    data, normalized_embedding, samples, return_uncertainty
                )
            else:
                return self.forward_uncertainty_aware(
                    data, normalized_embedding, samples, return_uncertainty
                )


class UncertaintyAwareFullyMonteCarloIntegratedAttentionVariationalBatchGAT(nn.Module):
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
        attention_filter_limit=0.5,
        variational_mode_on_inference=False,
    ):
        super(UncertaintyAwareFullyMonteCarloIntegratedAttentionVariationalBatchGAT, self).__init__()

        self.default_samples = samples
        self.test_samples = test_samples
        self.attention_filter_limit = attention_filter_limit
        self.variational_mode_on_inference = variational_mode_on_inference

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
        # n_units[0] += 64

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

    def forward_variational(
        self, data, normalized_embedding=None, samples=None, return_uncertainty=False
    ):

        if samples is None:
            if self.training:
                samples = self.default_samples
            else:
                samples = self.test_samples

        input_xs = []
        output_xs = []
        h_primes = []
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
            h_primes = []

            for s in range(samples):
                x = input_xs.pop(0)
                # y, att_y = gat_layer((x, adj))
                ((h_prime_means, attention_means), (h_prime_stds, attention_stds)) = gat_layer.raw_attention_step((x, adj))
                x = gat_layer.variational_output_step(((h_prime_means, attention_means), (h_prime_stds, attention_stds)))[
                    0
                ]  # bs x n_head x n x f_out
                output_xs.append(x)

            if i + 1 == self.n_layer:
                for q in range(len(output_xs)):
                    output_xs[q] = output_xs[q].mean(dim=1)
            else:
                for q in range(len(output_xs)):
                    output_xs[q] = (
                        output_xs[q].transpose(1, 2).contiguous().view(bs, n, -1)
                    )

        for q in range(len(output_xs)):
            output_xs[q] = F.log_softmax(output_xs[q], dim=-1)[:, -1, :]

        result_var, result = torch.var_mean(
            torch.stack(output_xs, dim=0), dim=0, unbiased=False
        )

        if return_uncertainty:

            return result, result_var, attentions
        else:
            return result

    def forward_uncertainty_aware(
        self, data, normalized_embedding=None, samples=None, return_uncertainty=False
    ):

        if samples is None:
            if self.training:
                samples = self.default_samples
            else:
                samples = self.test_samples

        input_xs = []
        output_xs = []
        h_primes = []
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
            h_primes = []
            raw_attentions = []

            for s in range(samples):
                x = input_xs.pop(0)
                ((h_prime_means, attention_means), (h_prime_stds, attention_stds)) = gat_layer.raw_attention_step((x, adj))
                attention = gat_layer.variational_attention_sample(attention_means, attention_stds)[0]
                h_primes.append((h_prime_means, h_prime_stds))
                # x = gat_layer.output_step((h_prime, attention))[0]  # bs x n_head x n x f_out
                # x, attention = gat_layer.all_steps((x, adj))

                if i not in all_attentions:
                    all_attentions[i] = []

                all_attentions[i].append(attention)
                raw_attentions.append((attention_means, attention_stds))
                # output_xs.append(x)

            att_var, att = torch.var_mean(
                torch.stack(all_attentions[i], dim=0), dim=0, unbiased=False
            )

            attention_filtered, attention_var_filtered = filter_attentions(att, att_var, self.attention_filter_limit)

            attentions[i] = (att, att_var)

            for s in range(samples):
                (h_prime_means, h_prime_stds) = h_primes.pop(0)
                (attention_means, attention_stds) = raw_attentions.pop(0)
                raw_attention_filtered, raw_attention_stds_filtered = filter_attentions(attention_means, att_var, self.attention_filter_limit)
                x = gat_layer.variational_output_step(((h_prime_means, raw_attention_filtered), (h_prime_stds, raw_attention_stds_filtered)))[
                    0
                ]  # bs x n_head x n x f_out
                # x = gat_layer.variational_output_step(((h_prime_means, attention_filtered), (h_prime_stds, attention_var_filtered)))[
                #     0
                # ]  # bs x n_head x n x f_out
                output_xs.append(x)

            if i + 1 == self.n_layer:
                for q in range(len(output_xs)):
                    output_xs[q] = output_xs[q].mean(dim=1)
            else:
                for q in range(len(output_xs)):
                    output_xs[q] = (
                        output_xs[q].transpose(1, 2).contiguous().view(bs, n, -1)
                    )

        for q in range(len(output_xs)):
            output_xs[q] = F.log_softmax(output_xs[q], dim=-1)[:, -1, :]

        result_var, result = torch.var_mean(
            torch.stack(output_xs, dim=0), dim=0, unbiased=False
        )

        if return_uncertainty:

            return result, result_var, attentions
        else:
            return result

    def forward(
        self, data, normalized_embedding=None, samples=None, return_uncertainty=False
    ):

        if self.training:
            return Exception("Training forbidden for this model")
        else:
            if self.variational_mode_on_inference:
                return self.forward_variational(
                    data, normalized_embedding, samples, return_uncertainty
                )
            else:
                return self.forward_uncertainty_aware(
                    data, normalized_embedding, samples, return_uncertainty
                )
