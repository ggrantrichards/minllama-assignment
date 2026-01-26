from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    _, seqlen, _, _ = query.shape
    device = query.device

    # 1. Reshape to separate real/imag parts: (bs, seqlen, heads, head_dim/2, 2)
    # This "Interleaved" logic is correct for llama2.c / stories42M.pt
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # 2. Compute frequencies
    # i = [0, 2, ..., head_dim-2]
    i = torch.arange(0, head_dim, 2, device=device).float()
    
    # Calculate inv_freq using the exact llama2.c formula
    inv_freq = 1.0 / (theta ** (i / head_dim))
    
    positions = torch.arange(seqlen, device=device).float()
    
    # Outer product: (seqlen, head_dim/2)
    freqs_cis = torch.outer(positions, inv_freq)
    
    cos = torch.cos(freqs_cis)
    sin = torch.sin(freqs_cis)

    # 3. Broadcast dimensions to match query_real
    cos = reshape_for_broadcast(cos, query_real)
    sin = reshape_for_broadcast(sin, query_real)

    # 4. Apply Rotation (Standard Complex Multiplication)
    query_out_real = query_real * cos - query_imag * sin
    query_out_imag = query_real * sin + query_imag * cos
    
    key_out_real = key_real * cos - key_imag * sin
    key_out_imag = key_real * sin + key_imag * cos

    # 5. Stack and flatten back to (bs, seqlen, heads, head_dim)
    query_out = torch.stack((query_out_real, query_out_imag), dim=-1).flatten(-2)
    key_out = torch.stack((key_out_real, key_out_imag), dim=-1).flatten(-2)

    return query_out.type_as(query), key_out.type_as(key)