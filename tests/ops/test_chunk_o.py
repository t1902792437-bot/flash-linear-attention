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
            q_h = q[i_b, :, i_h // (HV // H), :]  # [T, K]
            k_h = k[i_b, :, i_h // (HV // H), :]  # [T, K]
            v_h = v[i_b, :, i_h, :]  # [T, V]
            
            for i_t in range(NT):
                # chunk indices
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T)
                actual_BT = t_end - t_start
                
                # intra-chunk attention: Q @ K^T
                q_chunk = q_h[t_start:t_end]  # [BT, K]
                k_chunk = k_h[t_start:t_end]  # [BT, K]
                v_chunk = v_h[t_start:t_end]  # [BT, V]
                
                # A = Q @ K^T, causal mask
                A = torch.matmul(q_chunk, k_chunk.transpose(-1, -2))  # [BT, BT]
                causal_mask = torch.tril(torch.ones(actual_BT, actual_BT, device=q.device))
                A = A * causal_mask
                
                # inter-chunk contribution: Q @ h
                h_chunk = h[i_b, i_t, i_h // (HV // H)]  # [K, V]
                o_inter = torch.matmul(q_chunk, h_chunk)  # [BT, V]
                
                # apply gate if present
                if g is not None:
                    g_h = g[i_b, :, i_h].to(torch.float32)  # [T]
                    g_chunk = g_h[t_start:t_end]  # [BT]
                    # inter-chunk: multiply by exp(g)
                    o_inter = o_inter * torch.exp(g_chunk).unsqueeze(-1)
                    # intra-chunk: multiply by exp(g_i - g_j)
                    g_diff = g_chunk.unsqueeze(-1) - g_chunk.unsqueeze(-2)  # [BT, BT]
                    A = A * torch.exp(g_diff)
                
                # final output: scale * (inter + intra)
                o_intra = torch.matmul(A, v_chunk)  # [BT, V]
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
    
    # compute h using chunk_fwd_h
    h, _ = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        chunk_size=chunk_size,
    )
    
    # reference output
    ref = naive_chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        scale=scale,
        chunk_size=chunk_size,
    )
    
    # triton output
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


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'V', 'chunk_size', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-V{}-chunk{}-{}".format(*test))
        for test in [
            (1, 128, 4, 64, 64, 64, torch.float32),
            (2, 256, 8, 64, 128, 64, torch.float32),
            (1, 64, 2, 32, 64, 32, torch.float16),
            (2, 512, 8, 64, 128, 64, torch.float16),
        ]
    ],
)
def test_chunk_fwd_o_gqa(
    B: int,
    T: int,
    H: int,
    D: int,
    V: int,
    chunk_size: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    
    # GQA: H_q > H_v
    HV = H // 2  # half the heads for values
    
    q = torch.randn((B, T, H, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, HV, V), dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn((B, T, HV), dtype=dtype, device=device))
    
    scale = D ** -0.5
    
    # compute h
    h, _ = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        chunk_size=chunk_size,
    )
    
    # naive reference (need to handle GQA)
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    h_f = h.to(torch.float32)
    g_f = g.to(torch.float32) if g is not None else None
    
    NT = (T + chunk_size - 1) // chunk_size
    ref = torch.zeros(B, T, HV, V, dtype=torch.float32, device=device)
    
    for i_b in range(B):
        for i_hv in range(HV):
            i_hq = i_hv // (HV // H) * (H // HV) + i_hv % (H // HV)
            # simple mapping: each v head corresponds to multiple q heads
            # actually for GQA, we need proper head mapping
            q_h = q_f[i_b, :, i_hv * (H // HV), :]  # pick first q head for this v head
            k_h = k_f[i_b, :, i_hv * (H // HV), :]
            v_h = v_f[i_b, :, i_hv, :]
            g_h = g_f[i_b, :, i_hv] if g_f is not None else None
            
            for i_t in range(NT):
                t_start = i_t * chunk_size
                t_end = min((i_t + 1) * chunk_size, T)
                BT = t_end - t_start
                
                q_chunk = q_h[t_start:t_end]
                k_chunk = k_h[t_start:t_end]
                v_chunk = v_h[t_start:t_end]
                
                A = torch.matmul(q_chunk, k_chunk.transpose(-1, -2))
                causal_mask = torch.tril(torch.ones(BT, BT, device=device))
                A = A * causal_mask
                
                h_chunk = h_f[i_b, i_t, i_hv]
                o_inter = torch.matmul(q_chunk, h_chunk)
                
                if g_h is not None:
                    g_chunk = g_h[t_start:t_end]
                    o_inter = o_inter * torch.exp(g_chunk).unsqueeze(-1)
                    g_diff = g_chunk.unsqueeze(-1) - g_chunk.unsqueeze(-2)
                    A = A * torch.exp(g_diff)
                
                o_intra = torch.matmul(A, v_chunk)
                ref[i_b, t_start:t_end, i_hv, :] = scale * (o_inter + o_intra)
    
    # triton output
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


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'chunk_size', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-chunk{}-{}".format(*test))
        for test in [
            (1, 64, 1, 64, 64, torch.float32),
            (2, 128, 4, 64, 64, torch.float32),
            (2, 256, 4, 64, 64, torch.float16),
            (1, 512, 8, 64, 64, torch.float16),
        ]
    ],
)
def test_chunk_fwd_o_with_h0(
    B: int,
    T: int,
    H: int,
    D: int,
    chunk_size: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    
    q = torch.randn((B, T, H, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn((B, T, H), dtype=dtype, device=device))
    h0 = torch.randn((B, H, D, D), dtype=torch.float32, device=device)
    
    scale = D ** -0.5
    
    # compute h with initial state
    h, _ = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        h0=h0,
        chunk_size=chunk_size,
    )
    
    # naive reference with h0
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    h_f = h.to(torch.float32)
    g_f = g.to(torch.float32)
    h0_f = h0.to(torch.float32)
    
    NT = (T + chunk_size - 1) // chunk_size
    ref = torch.zeros(B, T, H, D, dtype=torch.float32, device=device)
    
    for i_b in range(B):
        for i_h in range(H):
            q_h = q_f[i_b, :, i_h, :]
            k_h = k_f[i_b, :, i_h, :]
            v_h = v_f[i_b, :, i_h, :]
            g_h = g_f[i_b, :, i_h]
            
            # h0 is the initial state for this batch and head
            h0_bh = h0_f[i_b, i_h]  # [D, D]
            
            for i_t in range(NT):
                t_start = i_t * chunk_size
                t_end = min((i_t + 1) * chunk_size, T)
                BT = t_end - t_start
                
                q_chunk = q_h[t_start:t_end]
                k_chunk = k_h[t_start:t_end]
                v_chunk = v_h[t_start:t_end]
                
                A = torch.matmul(q_chunk, k_chunk.transpose(-1, -2))
                causal_mask = torch.tril(torch.ones(BT, BT, device=device))
                A = A * causal_mask
                
                # h contains the state at the beginning of each chunk
                # first chunk uses h0, others use accumulated state
                if i_t == 0:
                    h_chunk = h0_bh
                else:
                    h_chunk = h_f[i_b, i_t, i_h]
                
                o_inter = torch.matmul(q_chunk, h_chunk)
                
                g_chunk = g_h[t_start:t_end]
                o_inter = o_inter * torch.exp(g_chunk).unsqueeze(-1)
                g_diff = g_chunk.unsqueeze(-1) - g_chunk.unsqueeze(-2)
                A = A * torch.exp(g_diff)
                
                o_intra = torch.matmul(A, v_chunk)
                ref[i_b, t_start:t_end, i_h, :] = scale * (o_inter + o_intra)
    
    # triton output
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


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'chunk_size', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-chunk{}-{}".format(*test))
        for test in [
            (1, 127, 1, 64, 64, torch.float32),
            (2, 255, 4, 64, 64, torch.float32),
            (1, 63, 2, 32, 64, torch.float16),
            (2, 511, 8, 64, 64, torch.float16),
        ]
    ],
)
def test_chunk_fwd_o_non_divisible(
    B: int,
    T: int,
    H: int,
    D: int,
    chunk_size: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    
    # T is not divisible by chunk_size
    q = torch.randn((B, T, H, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn((B, T, H), dtype=dtype, device=device))
    
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