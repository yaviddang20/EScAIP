from tests.test_utils import load_data_model
import torch


def test_output_block():
    batch, model = load_data_model()

    max_nei = model.molecular_graph_cfg.max_neighbors
    hidden_size = model.global_cfg.hidden_size

    x = model.data_preprocess(batch)
    N = x.node_padding_mask.shape[0]

    edge_readout = torch.randn(
        (N, max_nei, hidden_size * (model.gnn_cfg.num_layers + 1)),
        device=batch.pos.device,
    )
    node_readout = torch.randn(
        (N, hidden_size * (model.gnn_cfg.num_layers + 1)), device=batch.pos.device
    )

    energy_output, force_output = model.output_block(
        node_readouts=node_readout,
        edge_readouts=edge_readout,
        edge_direction=x.edge_direction,
        node_batch=x.node_batch,
        neighbor_mask=x.neighbor_mask,
        num_graphs=batch.num_graphs + 1,
    )

    assert energy_output.shape == (batch.num_graphs + 1, 1)
    assert force_output.shape == (N, 3)
