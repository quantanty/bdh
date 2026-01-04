# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math
from typing import Iterable, Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V, cache_method=None, layer_state=None, t_offset=0):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        D = self.config.n_embd
        B, H, T, N = K.size()
        if cache_method == 'state' and layer_state is None:
            layer_state = torch.zeros((B, H, N, D), device=K.device)

        r_phases = (
            torch.arange(
                t_offset,
                t_offset + T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR

        # Current attention
        a = (QR @ KR.mT).tril(diagonal=-1) @ V
        if layer_state is not None:
            a += QR @ layer_state
            if cache_method == 'state':
                layer_state += KR.mT @ V.expand(-1, H, -1, -1)
        return a, layer_state


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, cache_method=None, state=None, t_offset=0):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh


        v = self.embed(idx).unsqueeze(1)

        # actually helps with training
        v = self.ln(v)  # B, 1, T, D

        for level in range(C.n_layer):
            x_sparse = F.relu(v @ self.encoder)  # B, nh, T, N

            if state is not None:
                y, state[level] = self.attn(
                    Q=x_sparse,
                    K=x_sparse,
                    V=v,
                    cache_method=cache_method,
                    layer_state=state[level],
                    t_offset=t_offset
                )
            else:
                y, _ = self.attn(
                    Q=x_sparse,
                    K=x_sparse,
                    V=v,
                    cache_method=cache_method,
                )
            y = self.ln(y)

            y = y @ self.encoder_v
            y = F.relu(y)
            y = x_sparse * y  # B, nh, T, N

            y = self.drop(y)

            y = (
                y.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(y)
            v = self.ln(v + y)

        logits = v.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, state

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        cache_method: Literal['state'] | None = None, # TODO: kv cache?
        state: Iterable | None = None,
        t_offset: int = 0
    ) -> tuple[torch.Tensor, Iterable | None, int]:
        self.eval()
        B, T = idx.size()
        if state is not None:
            for layer_state in state:
                assert layer_state.size(0) == B
        elif cache_method == 'state':
            state = [None] * self.config.n_layer
        
        feed = idx
        t0 = t_offset
        
        for _ in range(max_new_tokens):
            logits, _, state = self(
                feed,
                cache_method=cache_method,
                state=state,
                t_offset=t_offset
            )
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
            if cache_method is None:
                feed = idx
            else:
                feed = idx_next
                t_offset = t0 + idx.size(1)
                
        self.train()
        return idx, state, t_offset


if __name__ == "__main__":
    model = BDH(BDHConfig())
