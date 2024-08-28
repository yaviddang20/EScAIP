import torch
from xformers.ops.fmha import BlockDiagonalMask


def attn_bias_for_memory_efficient_attention(neighbor_mask) -> BlockDiagonalMask:
    """
    Generate attention bias for memory efficient attention.
    """
    # construct sequence length for block diagonal mask
    neighbor_count = neighbor_mask.sum(dim=-1)
    max_neighbors = neighbor_mask.shape[1]
    padding_size = max_neighbors - neighbor_count
    seq_lens = torch.stack([neighbor_count, padding_size], dim=1).view(-1).tolist()
    atten_bias = BlockDiagonalMask.from_seqlens(seq_lens)

    return atten_bias


@torch.compile
def attn_bias_for_scaled_dot_product(
    neighbor_mask: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """
    Generate attention bias for scaled dot product attention.
    """
    batch_size, max_neighbors = neighbor_mask.shape
    neighbor_count_expand = neighbor_mask.sum(dim=-1).view(batch_size, 1, 1)

    indices = torch.arange(max_neighbors, device=neighbor_mask.device).unsqueeze(0)
    row_indices = indices.unsqueeze(1).expand(batch_size, max_neighbors, max_neighbors)
    col_indices = indices.unsqueeze(2).expand(batch_size, max_neighbors, max_neighbors)

    atten_bias = torch.where(
        (row_indices < neighbor_count_expand) & (col_indices < neighbor_count_expand),
        torch.tensor(0.0),
        torch.tensor(float("-inf")),
    )

    # repeat for heads
    atten_bias = atten_bias.unsqueeze(1).expand(-1, num_heads, -1, -1)  # (B, H, S, S)
    atten_bias = atten_bias.flatten(start_dim=0, end_dim=1)

    return atten_bias
