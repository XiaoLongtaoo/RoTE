# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.gpt2 import modeling_gpt2 as gpt2mod
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from models.rpg.baseline import RPG
from modules import MultiLevelRoTE, RotaryTimeEmbedding


class GPT2AttentionWithRoTE(GPT2Attention):
    """GPT-2 attention with shared RoTE modules injected into Q/K."""

    def __init__(
        self,
        config,
        is_cross_attention: bool = False,
        layer_idx: int | None = None,
        rope_base: float = 1_000_000.0,
        rote_mode: str = "unix",
        year_base: float = 1_000_000.0,
        month_base: float = 10_000.0,
        day_base: float = 100.0,
        year_weight: float = 1.5,
        month_weight: float = 1.0,
        day_weight: float = 0.5,
    ):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.rote_mode = rote_mode
        if rote_mode == "unix":
            self.rote = RotaryTimeEmbedding(dim=self.head_dim, base=rope_base)
        elif rote_mode == "ymd":
            self.rote = MultiLevelRoTE(
                dim=self.head_dim,
                year_base=year_base,
                month_base=month_base,
                day_base=day_base,
                year_weight=year_weight,
                month_weight=month_weight,
                day_weight=day_weight,
            )
        else:
            raise ValueError(f"Unsupported rote_mode: {rote_mode}")

        self._time_ids: torch.Tensor | None = None
        self._year_ids: torch.Tensor | None = None
        self._month_ids: torch.Tensor | None = None
        self._day_ids: torch.Tensor | None = None

    def set_time_ids(self, time_ids: torch.Tensor | None):
        self.set_time_features(time_ids=time_ids)

    def set_time_features(
        self,
        time_ids: torch.Tensor | None = None,
        year_ids: torch.Tensor | None = None,
        month_ids: torch.Tensor | None = None,
        day_ids: torch.Tensor | None = None,
    ):
        self._time_ids = time_ids
        self._year_ids = year_ids
        self._month_ids = month_ids
        self._day_ids = day_ids

    def forward(
        self,
        hidden_states,
        past_key_value=None,
        cache_position=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        if past_key_value is not None:
            if isinstance(past_key_value, getattr(gpt2mod, "EncoderDecoderCache", tuple())):
                past_key_value = (
                    past_key_value.cross_attention_cache if is_cross_attention else past_key_value.self_attention_cache
                )
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs=cache_kwargs,
            )

        if not is_cross_attention:
            q_len = query_states.size(-2)
            k_len = key_states.size(-2)
            max_len = max(q_len, k_len)

            if self.rote_mode == "unix" and self._time_ids is not None:
                pos = self._time_ids.to(query_states.device)
                if pos.dim() != 2:
                    raise ValueError(f"time_ids must be (batch, seq_len), got {tuple(pos.shape)}")
                if pos.size(1) < max_len:
                    pad = max_len - pos.size(1)
                    pos = torch.cat([pos.new_zeros((pos.size(0), pad)), pos], dim=1)
                q_pos = pos[:, -q_len:].to(dtype=query_states.dtype)
                k_pos = pos[:, -k_len:].to(dtype=key_states.dtype)
                q_pos = q_pos - q_pos[:, :1]
                k_pos = k_pos - k_pos[:, :1]
                query_states, key_states = self.rote(query_states, key_states, q_pos=q_pos, k_pos=k_pos)

            elif self.rote_mode == "ymd" and self._year_ids is not None:
                y = self._year_ids.to(query_states.device)
                m = self._month_ids.to(query_states.device)
                d = self._day_ids.to(query_states.device)
                if y.dim() != 2:
                    raise ValueError(f"year_ids must be (batch, seq_len), got {tuple(y.shape)}")
                if y.size(1) < max_len:
                    pad = max_len - y.size(1)
                    y = torch.cat([y.new_zeros((y.size(0), pad)), y], dim=1)
                    m = torch.cat([m.new_zeros((m.size(0), pad)), m], dim=1)
                    d = torch.cat([d.new_zeros((d.size(0), pad)), d], dim=1)

                qy = y[:, -q_len:].to(dtype=query_states.dtype)
                qm = m[:, -q_len:].to(dtype=query_states.dtype)
                qd = d[:, -q_len:].to(dtype=query_states.dtype)
                ky = y[:, -k_len:].to(dtype=key_states.dtype)
                km = m[:, -k_len:].to(dtype=key_states.dtype)
                kd = d[:, -k_len:].to(dtype=key_states.dtype)

                qy = qy - qy[:, :1]
                qm = qm - qm[:, :1]
                qd = qd - qd[:, :1]
                ky = ky - ky[:, :1]
                km = km - km[:, :1]
                kd = kd - kd[:, :1]

                query_states, key_states = self.rote(
                    query_states,
                    key_states,
                    q_ymd=(qy, qm, qd),
                    k_ymd=(ky, km, kd),
                )

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention
        attn_impl = getattr(self.config, "_attn_implementation", "eager") or "eager"
        using_eager = attn_impl == "eager"
        attention_interface = gpt2mod.eager_attention_forward
        if attn_impl != "eager":
            if attn_impl == "sdpa" and (output_attentions or head_mask is not None):
                using_eager = True
                gpt2mod.logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                    'Falling back to eager attention. Set `attn_implementation="eager"` to remove this warning.'
                )
            else:
                attention_interface = gpt2mod.ALL_ATTENTION_FUNCTIONS[attn_impl]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, attn_weights


class RoTERPG(RPG):
    def __init__(self, config: dict, dataset, tokenizer):
        super().__init__(config, dataset, tokenizer)

        with torch.no_grad():
            self.gpt2.wpe.weight.zero_()
        self.gpt2.wpe.weight.requires_grad = False

        rote_mode = self.config.get("rote_mode", "unix")
        rope_base = float(self.config.get("year_base", 1_000_000.0))
        year_base = float(self.config.get("year_base", 1_000_000.0))
        month_base = float(self.config.get("month_base", 10_000.0))
        day_base = float(self.config.get("day_base", 100.0))

        for index, block in enumerate(self.gpt2.h):
            block.attn = GPT2AttentionWithRoTE(
                self.gpt2.config,
                layer_idx=index,
                rope_base=rope_base,
                rote_mode=rote_mode,
                year_base=year_base,
                month_base=month_base,
                day_base=day_base,
                year_weight=float(self.config.get("year_weight", 1.5)),
                month_weight=float(self.config.get("month_weight", 1.0)),
                day_weight=float(self.config.get("day_weight", 0.5)),
            )

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        input_ids = self._to_tensor(batch["input_ids"])
        attention_mask = self._to_tensor(batch["attention_mask"])
        input_tokens = self.item_id2tokens[input_ids]
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)

        if self.config.get("rote_mode", "unix") == "ymd":
            year_ids = batch.get("year_ids")
            month_ids = batch.get("month_ids")
            day_ids = batch.get("day_ids")
            if year_ids is None or month_ids is None or day_ids is None:
                raise ValueError("rote_mode=ymd but batch is missing year_ids/month_ids/day_ids.")
            year_ids = self._to_tensor(year_ids, dtype=torch.float32)
            month_ids = self._to_tensor(month_ids, dtype=torch.float32)
            day_ids = self._to_tensor(day_ids, dtype=torch.float32)
            for block in self.gpt2.h:
                block.attn.set_time_features(
                    time_ids=None,
                    year_ids=year_ids,
                    month_ids=month_ids,
                    day_ids=day_ids,
                )
        else:
            time_ids = batch.get("time_ids")
            time_ids = self._to_tensor(time_ids, dtype=torch.float32) if time_ids is not None else None
            for block in self.gpt2.h:
                block.attn.set_time_ids(time_ids)

        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            position_ids=torch.zeros_like(input_ids),
        )
        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        outputs.final_states = final_states
        if return_loss:
            if "labels" not in batch:
                raise AssertionError("The batch must contain the labels.")
            labels = self._to_tensor(batch["labels"])
            label_mask = labels.view(-1) != -100
            selected_states = final_states.view(-1, self.n_pred_head, self.config["n_embd"])[label_mask]
            selected_states = F.normalize(selected_states, dim=-1)
            selected_states = torch.chunk(selected_states, self.n_pred_head, dim=1)
            token_emb = self.gpt2.wte.weight[1:-1]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
            token_logits = [
                torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / self.temperature
                for i in range(self.n_pred_head)
            ]
            token_labels = self.item_id2tokens[labels.view(-1)[label_mask]]
            losses = [
                self.loss_fct(token_logits[i], token_labels[:, i] - i * self.config["codebook_size"] - 1)
                for i in range(self.n_pred_head)
            ]
            outputs.loss = torch.mean(torch.stack(losses))
        return outputs
