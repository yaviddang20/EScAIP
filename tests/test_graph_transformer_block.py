from tests.test_utils import load_data_model


def test_graph_transformer_block():
    batch, model = load_data_model()

    max_nei = model.molecular_graph_cfg.max_neighbors
    hidden_size = model.global_cfg.hidden_size

    x = model.data_preprocess(batch)
    N = x.node_padding_mask.shape[0]

    node_features, edge_features = model.input_block(x)

    output = model.transformer_blocks[0](x, node_features, edge_features)

    assert output[0].shape == (N, hidden_size)
    assert output[1].shape == (N, max_nei, hidden_size)
