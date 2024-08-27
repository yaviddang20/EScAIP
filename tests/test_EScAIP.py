from tests.test_utils import load_data_model


def test_data_preprocess():
    batch, model = load_data_model()

    output = model(batch)

    assert output["energy"].shape == (batch.natoms.shape[0], 1)
    assert output["forces"].shape == (batch.num_nodes, 3)
