import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1 - self.p
        noise = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        mask = (noise >= self.p).to(x.dtype)
        return x / keep_prob * mask


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: Optional[int] = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg(x)
        w = self.fc(w)
        return x * w


class SEResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.conv = ConvBNReLU(channels, channels, kernel_size, stride=stride)
        self.se = SqueezeExcite(channels, reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.se(out)
        out = out + identity
        return self.relu(out)


class MultiScaleCNN(nn.Module):
    """
    并行三尺度实现:
    Stem: Conv7 -> BN -> ReLU -> MaxPool
    三条路径（kernel=3/5/7），每条路径由 [1x1升维 -> SE-ResBlock(kernel)] x3 组成，
    最终在通道维拼接，out_channels = 256 * 3
    输入: (B, C, T) -> 输出: (B, C_out, T')
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 三条尺度路径
        self.path3 = self._make_path(64, [64, 128, 256], kernel=3)
        self.path5 = self._make_path(64, [64, 128, 256], kernel=5)
        self.path7 = self._make_path(64, [64, 128, 256], kernel=7)

        self.out_channels = 256 * 3  # 拼接三条路径的通道数

    def _make_path(self, in_ch, channels_list, kernel):
        layers = []
        for out_ch in channels_list:
            layers.append(ConvBNReLU(in_ch, out_ch, kernel_size=1))  # 升维/匹配通道
            layers.append(SEResBlock(out_ch, kernel_size=kernel))
            in_ch = out_ch  # 下一 stage 输入通道
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        out3 = self.path3(x)
        out5 = self.path5(x)
        out7 = self.path7(x)
        out = torch.cat([out3, out5, out7], dim=1)
        return out


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bias_table = nn.Parameter(torch.zeros(2 * max_distance - 1, num_heads))
        try:
            nn.init.trunc_normal_(self.bias_table, std=0.02)
        except Exception:
            nn.init.normal_(self.bias_table, std=0.02)

    def forward(self, seq_len: int):
        device = self.bias_table.device
        coords = torch.arange(seq_len, device=device)
        relative = coords[None, :] - coords[:, None] + self.max_distance - 1
        relative = relative.clamp(0, 2 * self.max_distance - 2)
        bias = self.bias_table[relative]  # (L, L, H)
        return bias.permute(2, 0, 1)     # (H, L, L)


class TransformerEncoderLayer(nn.Module):
    """
    带相对位置偏置和 DropPath 的简单多头自注意力层
    输入/输出: (B, L, C)
    """
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float, drop_path: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rel_pos = RelativePositionBias(num_heads)
        self.attn_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x, src_mask=None):
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,L,L)
        bias = self.rel_pos(L).unsqueeze(0).to(attn_scores.device)        # (1,H,L,L)
        attn_scores = attn_scores + bias

        if src_mask is not None:
            if src_mask.dim() == 2:
                mask = (~src_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
                attn_scores = attn_scores.masked_fill(mask.expand(-1, self.num_heads, L, -1), float("-inf"))
            else:
                attn_scores = attn_scores + src_mask

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        attn_out = self.out_proj(context)

        x = x + self.drop_path(attn_out)
        x = self.norm1(x)
        x = x + self.drop_path(self.ffn(x))
        x = self.norm2(x)
        return x


class TransformerStack(nn.Module):
    def __init__(self, depth: int, d_model: int, num_heads: int, dim_feedforward: int, dropout: float, drop_path_rate: float):
        super().__init__()
        drop_rates = [float(r) for r in torch.linspace(0, drop_path_rate, steps=depth)]
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, drop_rates[i])
            for i in range(depth)
        ])

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class CRF(nn.Module):
    """
    Batch CRF:
    - forward(emissions, tags, mask): returns mean negative log-likelihood
    - decode(emissions, mask): viterbi paths (list of lists)
    emissions: (B, L, C), tags: (B, L), mask: (B, L) bool
    """
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask):
        log_den = self._compute_log_partition(emissions, mask)
        log_num = self._score_sentence(emissions, tags, mask)
        return log_den - log_num

    def decode(self, emissions, mask):
        return self._viterbi_decode(emissions, mask)

    def _compute_log_partition(self, emissions, mask):
        B, L, C = emissions.size()
        score = self.start_transitions + emissions[:, 0]
        for i in range(1, L):
            broadcast_score = score.unsqueeze(2)
            broadcast_em = emissions[:, i].unsqueeze(1)
            transition_scores = broadcast_score + self.transitions.unsqueeze(0) + broadcast_em
            score = torch.logsumexp(transition_scores, dim=1)
            # mask handling: if masked position, keep previous score (no update)
            mask_i = mask[:, i].unsqueeze(1)
            score = torch.where(mask_i, score, score)
        return torch.logsumexp(score + self.end_transitions, dim=1).mean()

    def _score_sentence(self, emissions, tags, mask):
        B, L, C = emissions.size()
        score = self.start_transitions[tags[:, 0]] + emissions[torch.arange(B), 0, tags[:, 0]]
        for i in range(1, L):
            t_i = tags[:, i]
            t_prev = tags[:, i - 1]
            transition_score = self.transitions[t_prev, t_i]
            emission_score = emissions[torch.arange(B), i, t_i]
            score = score + (transition_score + emission_score) * mask[:, i]
        last_tags = tags.gather(1, mask.sum(dim=1).long().unsqueeze(1) - 1).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score.mean()

    def _viterbi_decode(self, emissions, mask):
        B, L, C = emissions.size()
        score = self.start_transitions + emissions[:, 0]
        history = []
        for i in range(1, L):
            broadcast_score = score.unsqueeze(2)
            next_score = broadcast_score + self.transitions.unsqueeze(0)
            best_score, best_path = next_score.max(dim=1)
            best_score = best_score + emissions[:, i]
            mask_i = mask[:, i].unsqueeze(1)
            score = torch.where(mask_i, best_score, score)
            history.append(best_path)
        score = score + self.end_transitions
        _, last_tag = score.max(dim=1)
        seq_ends = mask.long().sum(dim=1) - 1
        paths = []
        for b in range(B):
            last = last_tag[b].item()
            path = [last]
            for hist in reversed(history[: seq_ends[b]]):
                last = hist[b, last].item()
                path.append(last)
            path.reverse()
            paths.append(path)
        return paths


@dataclass
class ModelConfig:
    in_channels: int = 1
    d_model: int = 256
    num_heads: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    num_transformer_layers: int = 3
    num_classes: int = 5
    seq_len: int = 30


class SleepStageModel(nn.Module):
    """
    MultiScaleCNN -> proj -> Transformer x3 (相对位置 + DropPath) -> 分类 -> CRF序列头
    输出阶段按 seq_len (默认30) 对应的时间步长。
    输入: (B, in_channels, T) -> 输出: emissions (B, L, num_classes), decoded paths
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = MultiScaleCNN(cfg.in_channels)
        self.proj = nn.Linear(self.backbone.out_channels, cfg.d_model)
        self.transformer = TransformerStack(
            depth=cfg.num_transformer_layers,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            drop_path_rate=cfg.drop_path_rate
        )
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)
        self.crf = CRF(cfg.num_classes)

    def forward_features(self, x):
        # x: (B, C, T) -> (B, T', d_model)
        feats = self.backbone(x)                 # (B, C_out, T')
        feats = feats.transpose(1, 2)            # (B, T', C_out)
        feats = self.proj(feats)                 # (B, T', d_model)
        feats = self.transformer(feats)          # (B, T', d_model)
        return feats

    def forward(self, x, mask=None):
        feats = self.forward_features(x)
        emissions = self.classifier(feats)       # (B, L, num_classes)
        if mask is None:
            mask = torch.ones(emissions.size()[:2], dtype=torch.bool, device=emissions.device)
        return emissions, self.crf.decode(emissions, mask)

    def loss(self, x, tags, mask):
        feats = self.forward_features(x)
        emissions = self.classifier(feats)
        return self.crf(emissions, tags, mask), emissions