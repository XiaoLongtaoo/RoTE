from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import MultiLevelRoTE, decompose_unix_timestamp


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        return outputs.transpose(-1, -2)


class RoPEMultiheadAttentionYMD(nn.Module):
    """Multi-head self-attention backed by the shared multi-level RoTE module."""

    def __init__(
        self,
        hidden_units: int,
        num_heads: int,
        dropout_rate: float,
        year_base: float,
        month_base: float,
        day_base: float,
        year_weight: float,
        month_weight: float,
        day_weight: float,
    ):
        super().__init__()
        if hidden_units % num_heads != 0:
            raise ValueError("hidden_units must be divisible by num_heads")

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads

        self.q_proj = nn.Linear(hidden_units, hidden_units)
        self.k_proj = nn.Linear(hidden_units, hidden_units)
        self.v_proj = nn.Linear(hidden_units, hidden_units)
        self.out_proj = nn.Linear(hidden_units, hidden_units)

        self.rope = MultiLevelRoTE(
            self.head_dim,
            year_base=year_base,
            month_base=month_base,
            day_base=day_base,
            year_weight=year_weight,
            month_weight=month_weight,
            day_weight=day_weight,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_ymd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        k_ymd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor | None = None,
        rope_cache=None,
    ):
        seq_len, batch_size, _ = query.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.permute(1, 0, 2).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.permute(1, 0, 2).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.permute(1, 0, 2).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if rope_cache is not None:
            q = self.rope.apply(q, rope_cache)
            k = self.rope.apply(k, rope_cache)
        else:
            q, k = self.rope(q, k, q_ymd, k_ymd)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_units)
        attn_output = attn_output.permute(1, 0, 2)
        return self.out_proj(attn_output), None


class RoTESASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self._cached_causal_mask = None
        self._cached_causal_mask_len = None

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                RoPEMultiheadAttentionYMD(
                    args.hidden_units,
                    args.num_heads,
                    args.dropout_rate,
                    year_base=args.year_base,
                    month_base=args.month_base,
                    day_base=args.day_base,
                    year_weight=args.year_weight,
                    month_weight=args.month_weight,
                    day_weight=args.day_weight,
                )
            )
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

    def _get_causal_mask(self, seqs: torch.Tensor) -> torch.Tensor:
        length = seqs.shape[1]
        if (
            self._cached_causal_mask is None
            or self._cached_causal_mask_len != length
            or self._cached_causal_mask.device != seqs.device
        ):
            self._cached_causal_mask = ~torch.tril(torch.ones((length, length), dtype=torch.bool, device=seqs.device))
            self._cached_causal_mask_len = length
        return self._cached_causal_mask

    def log2feats(self, log_seqs, time_seqs):
        seqs = self.item_emb(torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.emb_dropout(seqs)

        time_tensor = torch.as_tensor(time_seqs, dtype=torch.float32, device=self.dev)
        time_ymd = decompose_unix_timestamp(time_tensor)
        attention_mask = self._get_causal_mask(seqs)
        rope_cache = self.attention_layers[0].rope.precompute(time_ymd) if self.attention_layers else None

        for index, attention in enumerate(self.attention_layers):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[index](seqs)
                mha_outputs, _ = attention(
                    x,
                    x,
                    x,
                    q_ymd=time_ymd,
                    k_ymd=time_ymd,
                    attn_mask=attention_mask,
                    rope_cache=rope_cache,
                )
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[index](self.forward_layernorms[index](seqs))
            else:
                mha_outputs, _ = attention(
                    seqs,
                    seqs,
                    seqs,
                    q_ymd=time_ymd,
                    k_ymd=time_ymd,
                    attn_mask=attention_mask,
                    rope_cache=rope_cache,
                )
                seqs = self.attention_layernorms[index](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[index](seqs + self.forward_layers[index](seqs))

        return self.last_layernorm(seqs)

    def forward(self, user_ids, log_seqs, time_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs, time_seqs)
        pos_embs = self.item_emb(torch.as_tensor(pos_seqs, dtype=torch.long, device=self.dev))
        neg_embs = self.item_emb(torch.as_tensor(neg_seqs, dtype=torch.long, device=self.dev))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, time_seqs, item_indices):
        log_feats = self.log2feats(log_seqs, time_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.as_tensor(item_indices, dtype=torch.long, device=self.dev))
        return item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

