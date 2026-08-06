# Subset of KCE building blocks used by DAE-TSE inference.
# Training-only utilities (PIT losses, RNN/CNN blocks, SI-SDR, etc.) are omitted.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

norm_dict = {
    'LayerNorm': nn.LayerNorm,
    'BatchNorm1d': nn.BatchNorm1d,
    'BatchNorm2d': nn.BatchNorm2d,
    'BatchNorm3d': nn.BatchNorm3d,
    'InstanceNorm1d': nn.InstanceNorm1d,
    'InstanceNorm2d': nn.InstanceNorm2d,
    'InstanceNorm3d': nn.InstanceNorm3d,
}

act_dict = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
}

CE_IGNORE_INDEX = -100


class CTC(nn.Module):
    """Kept for checkpoint compatibility (asr_phn_criterion.*)."""

    def __init__(self, num_tokens, front_output_size, reduce=True):
        super(CTC, self).__init__()
        self.linear_project = nn.Linear(front_output_size, num_tokens)
        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = nn.CTCLoss(reduction=reduction_type)

    def forward(self, logit, label, hyp_len, label_len, return_hyp=False):
        logit = self.linear_project(logit)
        logit = logit.transpose(0, 1)
        hyp = logit.log_softmax(2)
        loss = self.ctc_loss(hyp, label, hyp_len, label_len)
        loss = loss / hyp.size(1)
        if return_hyp:
            return loss, hyp
        return loss

    @torch.no_grad()
    def get_hyp(self, logit):
        return self.linear_project(logit)


class CE(nn.Module):
    """Speaker pooling + CE head; forward_pooling is used at TSE inference time."""

    def __init__(
        self,
        num_classes: int,
        front_output_size: int,
        pooling_conf: dict,
        use_reg_loss: bool = True,
        reg_loss_weight: float = 0.01,
        reduce: bool = True,
    ):
        super(CE, self).__init__()
        self.linear_project = nn.Linear(front_output_size, num_classes)
        reduction_type = "sum" if reduce else "none"
        self.ce_loss = nn.CrossEntropyLoss(
            reduction=reduction_type, ignore_index=CE_IGNORE_INDEX)

        self.pooling_type = pooling_conf['type']
        self.layer_indices = torch.tensor(
            pooling_conf.get('layer_indices', []), dtype=torch.long)
        self.use_softmax = pooling_conf['use_softmax']
        self.use_reg_loss = use_reg_loss
        self.reg_loss_weight = reg_loss_weight
        if self.pooling_type == 'learnable_weights':
            self.weight = nn.Parameter(
                torch.tensor([
                    1 / len(self.layer_indices)
                    for _ in range(len(self.layer_indices))
                ]))

    def forward_pooling(self, logit_list: list, sph_len: torch.Tensor) -> torch.Tensor:
        device = logit_list[0].device
        logit = torch.stack(logit_list)
        logit = logit[self.layer_indices.to(device).long()]
        if self.pooling_type == 'mean':
            pooled_logit = logit.mean(dim=0)
        elif self.pooling_type == 'learnable_weights':
            if self.use_softmax:
                _weight = F.softmax(self.weight, dim=-1)
            else:
                _weight = self.weight
            pooled_logit = sum(w * v for (w, v) in zip(_weight, logit))

        pooled_logit = torch.stack([
            _[:length, :].mean(0) for (_, length) in zip(pooled_logit, sph_len)
        ])
        return pooled_logit

    def forward(self, logit_list: list, sph_len: torch.Tensor, label: torch.Tensor,
                is_target: torch.Tensor):
        pooled_logit = self.forward_pooling(logit_list, sph_len)
        pooled_logit = self.linear_project(pooled_logit)
        label[is_target == 0] = CE_IGNORE_INDEX
        loss = self.ce_loss(pooled_logit, label)
        loss = loss / pooled_logit.size(0)
        if self.use_reg_loss:
            loss += self.reg_loss_weight * (self.weight.norm(2) - 1) ** 2
        return loss


class WordEmbedding(nn.Module):
    def __init__(self, num_tokens, dim, padding_idx=0):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(num_tokens, dim, padding_idx=padding_idx)

    def forward(self, input):
        return self.emb(input)


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = model_dim
        self.xcale = math.sqrt(self.d_model)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        self.pe[0:, 0::2] = torch.sin(position * div_term)
        self.pe[0:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def position_encoding(self, size, device):
        index = torch.arange(0, size).to(device)
        return F.embedding(index, self.pe[0])

    def forward(self, input):
        self.pe = self.pe.to(input.device)
        pos_emb = self.position_encoding(input.size(1), input.device)
        return input * self.xcale + pos_emb


class MultiHeadAtt(nn.Module):
    def __init__(self, n_head, n_feats):
        super(MultiHeadAtt, self).__init__()
        assert n_feats % n_head == 0
        self.d_k = n_feats // n_head
        self.n_head = n_head
        self.q = nn.Linear(n_feats, n_feats)
        self.k = nn.Linear(n_feats, n_feats)
        self.v = nn.Linear(n_feats, n_feats)
        self.linear_out = nn.Linear(n_feats, n_feats)

    def forward_qkv(self, q, k, v):
        batch = q.size(0)
        q = self.q(q).view(batch, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.k(k).view(batch, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.v(v).view(batch, -1, self.n_head, self.d_k).transpose(1, 2)
        return q, k, v

    def forward_selfatt(self, q, k, v, mask, softmax=True):
        batch, nhead, tq, d = q.size()
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        if mask is not None:
            if len(mask.size()) < len(q.size()):
                mask = mask.unsqueeze(1).eq(0)
            else:
                mask = mask.eq(0)
            score = score.masked_fill(mask, -float('inf'))
            att_weight = torch.softmax(score, dim=-1).masked_fill(mask, 0.0)
        else:
            att_weight = torch.softmax(score, dim=-1)
        context = torch.matmul(att_weight, v)
        context = context.transpose(1, 2).contiguous().view(batch, tq, -1)
        return context, att_weight if softmax else score

    def forward(self, q, k, v, mask, softmax=True):
        q, k, v = self.forward_qkv(q, k, v)
        context, score = self.forward_selfatt(q, k, v, mask)
        return self.linear_out(context), score


class MultiHeadCrossAtt(nn.Module):
    def __init__(self, n_head, n_feats, norm=None):
        super(MultiHeadCrossAtt, self).__init__()
        self.q = nn.Sequential(norm_dict[norm](n_feats), nn.Linear(n_feats, n_feats))
        self.k = nn.Sequential(norm_dict[norm](n_feats), nn.Linear(n_feats, n_feats))
        self.v = nn.Sequential(norm_dict[norm](n_feats), nn.Linear(n_feats, n_feats))
        self.linear_out = nn.Linear(n_feats, n_feats)
        self.n_head = n_head
        self.n_feats = n_feats

    def forward_selfatt(self, q, k, v, mask, aux_score=None, softmax=True, print_mask=False):
        batch, nhead, tq, d = q.size()
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        if mask is not None:
            if len(mask.size()) < len(q.size()):
                mask = mask.unsqueeze(1).eq(0)
            else:
                mask = mask.eq(0)
        if aux_score is not None:
            score = score * aux_score[:, None, None, :]
        if mask is not None:
            score = score.masked_fill(mask, -float('inf'))
            att_weight = torch.softmax(score, dim=-1).masked_fill(mask, 0.0)
        else:
            att_weight = torch.softmax(score, dim=-1)
        context = torch.matmul(att_weight, v)
        context = context.transpose(1, 2).contiguous().view(batch, tq, -1)
        return context, att_weight if softmax else score

    def forward(self, q, k, v, mask=None, aux_score=None, softmax=True, print_mask=False):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        batch = k.size(0)
        tq, tk, tv = q.size(1), k.size(1), v.size(1)
        q = q.view(batch, tq, self.n_head, -1).transpose(1, 2)
        k = k.view(batch, tk, self.n_head, -1).transpose(1, 2)
        v = v.view(batch, tv, self.n_head, -1).transpose(1, 2)
        context, score = self.forward_selfatt(
            q, k, v, mask, aux_score=aux_score, softmax=softmax, print_mask=print_mask)
        return self.linear_out(context), score


class BaseConv(nn.Module):
    """Only ``compute_dim_reduction`` is used at inference time."""

    @staticmethod
    def compute_dim_reduction(input_dim, kernel, stride, padding, dilation, dim=0):
        p = padding if isinstance(padding, int) else padding[dim]
        d = dilation if isinstance(dilation, int) else dilation[dim]
        k = kernel if isinstance(kernel, int) else kernel[dim]
        s = stride if isinstance(stride, int) else stride[dim]
        idim = input_dim if not isinstance(input_dim, tuple) else input_dim[dim]
        rdim = idim + 2 * p - d * (k - 1) - 1
        rdim = torch.div(rdim, s, rounding_mode='floor') + 1
        return rdim


class DepthWiseConv(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=15,
        activation=nn.ReLU(),
        norm="batch_norm",
        causal=False,
        bias=True,
    ):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias)
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, stride=1, padding=padding,
            groups=channels, bias=bias)
        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.activation = activation

    def forward(self, input):
        input = input.transpose(1, 2)
        input = self.pointwise_conv1(input)
        input = nn.functional.glu(input, dim=1)
        input = self.depthwise_conv(input)
        if self.use_layer_norm:
            input = input.transpose(1, 2)
        input = self.activation(self.norm(input))
        if self.use_layer_norm:
            input = input.transpose(1, 2)
        input = self.pointwise_conv2(input)
        return input.transpose(1, 2)


class FNNBlock(nn.Module):
    def __init__(self, dim, num_block=1, bias=True, norm=None, act='ReLU'):
        super(FNNBlock, self).__init__()
        if isinstance(dim, list):
            assert len(dim) <= 3
            if len(dim) == 2:
                idim, hdim, odim = dim[0], None, dim[1]
            else:
                idim, hdim, odim = dim
        else:
            assert isinstance(dim, int)
            idim = hdim = odim = dim

        if norm is not None:
            assert norm in norm_dict
        if act is not None:
            assert act in act_dict

        self.w1 = nn.Linear(idim, hdim if hdim else idim)
        self.act = nn.ReLU()
        self.w2 = nn.Linear(hdim if hdim else idim, odim)

    def forward(self, input):
        return self.w2(self.act(self.w1(input)))


class TransformerLayer(nn.Module):
    def __init__(
        self,
        self_att,
        feed_forward,
        macaron_layer=None,
        conv_layer=None,
        cross_att=None,
        decoder_att=None,
        size=256,
    ):
        super(TransformerLayer, self).__init__()
        self.self_att = self_att
        self.feed_forward = feed_forward
        self.macaron_layer = macaron_layer
        self.cross_att = cross_att
        self.decoder_att = decoder_att
        self.input_norm = nn.LayerNorm(size, eps=1e-5)
        self.fnn_norm = nn.LayerNorm(size, eps=1e-5)
        self.size = size
        self.conv_layer = conv_layer
        if self.conv_layer is not None:
            self.conv_norm = nn.LayerNorm(size, eps=1e-5)
        if self.macaron_layer is not None:
            self.macaron_norm = nn.LayerNorm(size, eps=1e-5)
            self.macaron_factor = 0.5

    def forward(self, input, mask, cross_input=None, aux_score=None, args=None,
                print_mask=False):
        if self.macaron_layer is not None:
            residual = input
            input = self.macaron_norm(input)
            input = residual + self.macaron_factor * self.macaron_layer(input)

        residual = input
        input = self.input_norm(input)
        context, att_score = self.self_att(input, input, input, mask)
        input = residual + context
        residual = input

        if self.cross_att is not None:
            k, v, cross_mask = cross_input
            cross_context, cross_score = self.cross_att(
                input, k, v, cross_mask, aux_score)
            input = cross_context + residual
            residual = input

        if self.decoder_att is not None:
            memory, memory, memory_mask = cross_input
            decoder_context, decoder_score = self.decoder_att(
                input, memory, memory, memory_mask, print_mask=print_mask)
            input = decoder_context + residual
            residual = input

        if self.conv_layer is not None:
            conv_out = self.conv_norm(input)
            conv_out = self.conv_layer(conv_out)
            input = conv_out + residual
            residual = input

        input = self.fnn_norm(input)
        if args is not None:
            fnn_output, act = self.feed_forward(input, **args)
        else:
            fnn_output = self.feed_forward(input)
        input = fnn_output + residual

        if args is not None:
            return input, act
        elif self.cross_att is not None:
            return input, cross_score
        else:
            return input, att_score


def make_mask(length, max_len=None):
    assert isinstance(length, torch.Tensor)
    batch_size = length.size(0)
    max_len = max_len if max_len is not None else length.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=length.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = length.unsqueeze(-1)
    return seq_range_expand >= seq_length_expand


def combine_mask(mask1, mask2, tidx):
    t1 = mask1.size(tidx)
    t2 = mask2.size(tidx)
    mask1 = mask1.unsqueeze(1).repeat(1, t2, 1).transpose(-2, -1)
    mask2 = mask2.unsqueeze(1).repeat(1, t1, 1)
    cmask = mask1 & mask2
    return (~cmask).unsqueeze(1)
