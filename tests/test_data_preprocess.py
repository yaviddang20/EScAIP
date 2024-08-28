from tests.test_utils import load_data_model


def test_data_preprocess():
    batch, model = load_data_model()

    N = model.molecular_graph_cfg.max_num_nodes_per_batch * batch.num_graphs
    max_nei = model.molecular_graph_cfg.max_neighbors

    x = model.data_preprocess(batch)

    assert x.atomic_numbers.shape == (N,)
    assert x.node_direction_expansion.shape == (
        N,
        model.gnn_cfg.node_direction_expansion_size,
    )
    assert x.edge_distance_expansion.shape == (
        N,
        max_nei,
        model.gnn_cfg.edge_distance_expansion_size,
    )
    assert x.edge_direction.shape == (N, max_nei, 3)
    assert x.neighbor_list.shape == (N, max_nei)
    assert x.neighbor_mask.shape == (N, max_nei)
    assert x.node_batch.shape == (N,)
