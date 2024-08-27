import torch
import yaml

from src.EScAIP import EfficientlyScaledAttentionInteratomicPotential


def load_data_model():
    batch = torch.load("tests/data/OC20_batch_3.pt")
    with open("tests/data/L2_H4_64_.yml") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    model = EfficientlyScaledAttentionInteratomicPotential(
        num_atoms=0, bond_feat_dim=0, num_targets=0, **config["model"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return batch.to(device), model.to(device)
