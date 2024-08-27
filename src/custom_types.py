from typing import Union
import dataclasses

import torch
from xformers.ops.fmha import AttentionBias


@dataclasses.dataclass
class GraphAttentionData:
    """
    Custom dataclass for storing graph data for Graph Attention Networks
    atomic_numbers: (N)
    edge_distance_expansion: (N, max_nei, edge_distance_expansion_size)
    edge_direction: (N, max_nei, 3)
    node_direction_expansion: (N, node_direction_expansion_size)
    neighbor_list: (N, max_nei)
    neighbor_mask: (N, max_nei)
    node_batch: (N)
    node_padding_mask: (N)
    graph_padding_mask: (num_graphs)
    attn_bias: AttentionBias for xformers kernel
    """

    atomic_numbers: torch.Tensor
    edge_distance_expansion: torch.Tensor
    edge_direction: torch.Tensor
    node_direction_expansion: torch.Tensor
    neighbor_list: torch.Tensor
    neighbor_mask: torch.Tensor
    node_batch: torch.Tensor
    node_padding_mask: torch.Tensor
    graph_padding_mask: torch.Tensor
    attn_bias: Union[AttentionBias, None]
