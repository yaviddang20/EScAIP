import torch
from torch import nn

from ..configs import (
    GlobalConfigs,
    GraphNeuralNetworksConfigs,
    MolecularGraphConfigs,
    RegularizationConfigs,
)
from ..custom_types import GraphAttentionData
from ..utils.stochastic_depth import StochasticDepth, SkipStochasticDepth
from ..utils.nn_utils import (
    NormalizationType,
    get_normalization_layer,
    get_linear,
    get_feedforward,
)
from ..utils.graph_utils import map_sender_receiver_feature
from .base_block import BaseGraphNeuralNetworkLayer
from torch.distributions.normal import Normal

class EfficientGraphAttentionBlockMoE(nn.Module):
    def __init__(
        self,
        global_cfg,
        molecular_graph_cfg,
        gnn_cfg,
        reg_cfg,
        num_experts=8,
        k=2,
        noisy_gating=True,
    ):
        super().__init__()
        
        # Store configs
        self.global_cfg = global_cfg
        self.molecular_graph_cfg = molecular_graph_cfg
        self.gnn_cfg = gnn_cfg
        self.reg_cfg = reg_cfg
        
        # MoE parameters
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating
        # Create experts - each expert is a standard attention block
        self.experts = nn.ModuleList([
            EfficientGraphAttentionBlock(
                global_cfg=global_cfg,
                molecular_graph_cfg=molecular_graph_cfg,
                gnn_cfg=gnn_cfg,
                reg_cfg=reg_cfg
            ) for _ in range(num_experts)
        ])
        
        # Gating network parameters - using the same embedding dimension as EScAIP
        self.w_gate = nn.Parameter(torch.zeros(global_cfg.hidden_size, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(global_cfg.hidden_size, num_experts))
        
        # Gating functions
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        
        # Register buffers for noise
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.node_linear = self.get_node_linear(global_cfg, reg_cfg)

    def get_node_linear(
        self, global_cfg: GlobalConfigs, reg_cfg: RegularizationConfigs
    ):
        return get_linear(
            in_features=2 * global_cfg.hidden_size,
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

    def aggregate(self, edge_features, neighbor_mask):
        neighbor_count = neighbor_mask.sum(dim=1, keepdim=True) + 1e-5
        return (edge_features * neighbor_mask.unsqueeze(-1)).sum(dim=1) / neighbor_count
    
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)   
        return gates, load
    
    def get_node_features(
        self, node_features: torch.Tensor, neighbor_list: torch.Tensor
    ) -> torch.Tensor:
        sender_feature, receiver_feature = map_sender_receiver_feature(
            node_features, node_features, neighbor_list
        )
        return torch.cat([sender_feature, receiver_feature], dim=-1)
    

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, data: GraphAttentionData, node_features, edge_features):
        """
        Forward pass that maintains compatibility with EScAIP's data format
        Args:
            data: GraphAttentionData object containing graph structure
            node_features: Node features tensor
            edge_features: Edge features tensor
        """
        # Get gates for each node
        # tmp_node_feat = self.get_node_features(node_features, data.neighbor_list)
        # node_hidden = self.node_linear(tmp_node_feat)
        # print(tmp_node_feat.shape, node_hidden.shape)
        # exit()
        node_output = self.aggregate(edge_features, data.neighbor_mask)
        gates, load = self.noisy_top_k_gating(node_output, self.training)

        importance = gates.sum(0)
        # gates = self.noisy_top_k_gating(node_hidden.view(-1, self.global_cfg.hidden_size), self.training)
        # gates = gates.view(B, -1, self.num_experts)
        # Process through experts
        node_expert_outputs = []
        edge_expert_outputs = []
        for i in range(self.num_experts):
            # Each expert processes the same input format as original EScAIP
            node_expert_i_output, edge_expert_i_output = self.experts[i](data, node_features, edge_features)
            node_expert_outputs.append(node_expert_i_output)
            edge_expert_outputs.append(edge_expert_i_output)
        
        # Stack expert outputs
        node_expert_outputs = torch.stack(node_expert_outputs, dim=1)  # [B*num_nodes, num_experts, d_feature]
        edge_expert_outputs = torch.stack(edge_expert_outputs, dim=2)  # [B*num_nodes, num_experts, d_feature]
        
        # Combine expert outputs using gates
        node_features = gates.unsqueeze(dim=-1) * node_expert_outputs
        node_features = node_features.mean(dim=1)

        # edge_features = gates.unsqueeze(dim=-1) * edge_expert_outputs
        edge_features = edge_expert_outputs.mean(dim=2)

        # print(edge_features.shape)
        # exit()
        load_loss = (self.cv_squared(importance) + self.cv_squared(load))
        return node_features, edge_features, load_loss

class EfficientGraphAttentionBlock(nn.Module):
    """
    Efficient Graph Attention Block module.
    Ref: swin transformer
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        # Graph attention
        self.graph_attention = EfficientGraphAttention(
            global_cfg=global_cfg,
            molecular_graph_cfg=molecular_graph_cfg,
            gnn_cfg=gnn_cfg,
            reg_cfg=reg_cfg,
        )

        # Feed forward network
        self.feedforward = FeedForwardNetwork(
            global_cfg=global_cfg,
            gnn_cfg=gnn_cfg,
            reg_cfg=reg_cfg,
        )

        # Normalization
        normalization = NormalizationType(reg_cfg.normalization)
        self.norm_attn = get_normalization_layer(normalization, is_graph=False)(
            global_cfg.hidden_size
        )
        self.norm_ffn = get_normalization_layer(normalization)(global_cfg.hidden_size)

        # Stochastic depth
        self.stochastic_depth_attn = (
            StochasticDepth(reg_cfg.stochastic_depth_prob)
            if reg_cfg.stochastic_depth_prob > 0.0
            else SkipStochasticDepth()
        )
        self.stochastic_depth_ffn = (
            StochasticDepth(reg_cfg.stochastic_depth_prob)
            if reg_cfg.stochastic_depth_prob > 0.0
            else SkipStochasticDepth()
        )

    def forward(
        self,
        data: GraphAttentionData,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        # ref: swin transformer https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py#L452
        # x = x + self.stochastic_depth(self.graph_attention(self.norm_attn(x)))
        # x = x + self.stochastic_depth(self.feedforward(self.norm_ffn(x)))

        # attention
        node_hidden = self.norm_attn(node_features)
        node_hidden, edge_hidden = self.graph_attention(data, node_hidden)
        node_hidden, edge_hidden = self.stochastic_depth_attn(
            node_hidden, edge_hidden, data.node_batch
        )
        node_features, edge_features = (
            node_hidden + node_features,
            edge_hidden + edge_features,
        )

        # feedforward
        node_hidden, edge_hidden = self.norm_ffn(node_features, edge_features)
        node_hidden, edge_hidden = self.feedforward(node_hidden, edge_hidden)
        node_hidden, edge_hidden = self.stochastic_depth_ffn(
            node_hidden, edge_hidden, data.node_batch
        )
        node_features, edge_features = (
            node_hidden + node_features,
            edge_hidden + edge_features,
        )
        return node_features, edge_features


class EfficientGraphAttention(BaseGraphNeuralNetworkLayer):
    """
    Efficient Graph Attention module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__(global_cfg, molecular_graph_cfg, gnn_cfg, reg_cfg)

        # Edge linear layer
        self.edge_linear = self.get_edge_linear(gnn_cfg, global_cfg, reg_cfg)

        # Node hidden layer
        self.node_linear = self.get_node_linear(global_cfg, reg_cfg)

        # message linear
        self.message_linear = get_linear(
            in_features=global_cfg.hidden_size * 2,
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
        )

        # Multi-head attention
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=global_cfg.hidden_size,
            num_heads=gnn_cfg.atten_num_heads,
            dropout=reg_cfg.atten_dropout,
            bias=True,
            batch_first=True,
        )

        # scalar for attention bias
        self.use_angle_embedding = gnn_cfg.use_angle_embedding
        if self.use_angle_embedding:
            self.attn_scalar = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            self.attn_scalar = torch.tensor(1.0)

    def forward(
        self,
        data: GraphAttentionData,
        node_features: torch.Tensor,
    ):
        # Get edge features
        edge_features = self.get_edge_features(data)
        edge_hidden = self.edge_linear(edge_features)

        # Get node features
        node_features = self.get_node_features(node_features, data.neighbor_list)
        node_hidden = self.node_linear(node_features)

        # Concatenate edge and node features (num_nodes, num_neighbors, hidden_size)
        message = self.message_linear(torch.cat([edge_hidden, node_hidden], dim=-1))

        # Multi-head self-attention
        if self.use_angle_embedding:
            attn_mask = data.attn_mask + data.angle_embedding * self.attn_scalar
        else:
            attn_mask = data.attn_mask
        edge_output = self.multi_head_attention(
            query=message,
            key=message,
            value=message,
            # key_padding_mask=~data.neighbor_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )[0]

        # Aggregation
        node_output = self.aggregate(edge_output, data.neighbor_mask)

        return node_output, edge_output


class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()
        self.mlp_node = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.ffn_hidden_layer_multiplier,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )
        self.mlp_edge = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.ffn_hidden_layer_multiplier,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor):
        return self.mlp_node(node_features), self.mlp_edge(edge_features)
