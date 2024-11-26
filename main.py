import logging

import torch.optim as optim

import torch

from dataset.moon_dataset import MoonsDataset
from model.realnvp import RealNVP2D
from train import train_model

import argparse
import yaml

from utils.visualisation_tools import visualise_training


def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main_training(config):
    # log setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else
                          "cpu")

    dataset = MoonsDataset(**config["dataset"], device=device)
    realNVP = RealNVP2D(config["masks"], config["hidden_dim"]).to(device)

    optimiser = optim.Adam(realNVP.parameters(), lr=config["learning_rate"])

    losses = train_model(
        model=realNVP,
        dataset=dataset,
        optimiser=optimiser,
        num_steps=config["num_steps"],
        device=device,
        logger=logger,
    )

    checkpoint_path = config['checkpoint_path']
    torch.save(realNVP.state_dict(), checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")

    visualise_training(realNVP, losses, device)


if __name__ == '__main__':
    # the path to the config file
    parser = argparse.ArgumentParser(description="Load configuration for RealNVP training.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # load conf from the specified YAML file
    config = load_config(args.config)
    main_training(config)
