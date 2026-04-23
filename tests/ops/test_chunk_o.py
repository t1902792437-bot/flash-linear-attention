# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.common.chunk_h import chunk_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o
from fla.utils import assert_close, device


def naive_chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 64,
):
    B, T, H, K = q.shape
    V = v.shape[-1]
    HV = v.shape[2]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    
    if scale is None:
        scale = K ** -0.5
    
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    h = h.to(torch.float32)
    
    o = torch.zeros(B, T, HV, V, dtype=torch.float32, device=q.device)
    
    for i_b in range(B):
        for i_h in range(HV):
            q_h = q[i_b, :, i_h // (HV // H), :]
            k_h = k[i_b, :, i_h // (HV // H), :]
            v_h = v[i_b, :, i_h, :]
            
            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                actual_BT = t_end - t_start
                
                q_chunk = q_h[t_start:t_end]
                k_chunk = k_h[t_start:t_end]
                v_chunk = v_h[t_start:t_end]
                
                A = torch.matmul(q_chunk, k_chunk.transpose(-1, -2))
                causal_mask = torch.tril(torch.ones(actual_BT, actual_BT, device=q.device))
                A = A * causal_mask
                
                h_chunk = h[i_b, i_t, i_h // (HV // H)]
                o_inter = torch.matmul(q_chunk, h_chunk)
                
                if g is not None:
                    g_h = g[i_b, :, i_h].to(torch.float32)
                    g_chunk = g_h[t_start:t_end]
                    o_inter = o_inter * torch.exp(g_chunk).unsqueeze(-1)
                    g_diff = g_chunk.unsqueeze(-1) - g_chunk.unsqueeze(-2)
                    A = A * torch.exp(g_diff)
                
                o_intra = torch.matmul(A, v_chunk)
                o[i_b, t_start:t_end, i_h, :] = scale * (o_inter + o_intra)
    
    return o


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'V', 'chunk_size', 'use_gate', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-V{}-chunk{}-gate{}-{}".format(*test))
        for test in [
            (1, 64, 1, 64, 64, 64, False, torch.float32),
            (2, 128, 4, 64, 64, 64, False, torch.float32),
            (2, 256, 4, 64, 64, 64, True, torch.float32),
            (1, 128, 2, 32, 64, 32, False, torch.float32),
            (2, 512, 8, 64, 128, 64, True, torch.float32),
            (1, 256, 4, 64, 64, 64, False, torch.float16),
            (2, 512, 8, 64, 128, 64, True, torch.float16),
        ]
    ],
)
def test_chunk_fwd_o(
    B: int,
    T: int,
    H: int,
    D: int,
    V: int,
    chunk_size: int,
    use_gate: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    
    q = torch.randn((B, T, H, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, V), dtype=dtype, device=device)
    
    if use_gate:
        g = F.logsigmoid(torch.randn((B, T, H), dtype=dtype, device=device))
    else:
        g = None
    
    scale = D ** -0.5
    
    h, _ = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        chunk_size=chunk_size,
    )
    
    ref = naive_chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        scale=scale,
        chunk_size=chunk_size,
    )
    
    tri = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        scale=scale,
        chunk_size=chunk_size,
    )
    
    assert_close('o', ref, tri, 0.005)