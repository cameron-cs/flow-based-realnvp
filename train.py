import math

import torch

from dataset.moon_dataset import MoonsDataset
from model.realnvp import RealNVP2D


# Training Function
def train_model(model: RealNVP2D, dataset: MoonsDataset, optimiser, num_steps: int, device: str, logger):
    """
    Trains the RealNVP model on the specified dataset.

    Args:
        model: RealNVP model to train.
        dataset: MoonsDataset object for sampling data.
        optimiser: Optimiser for updating model parameters.
        num_steps: Total number of training steps.
        device: Device to run the model on ('cpu' or 'cuda').
        logger: Logger instance for logging progress.

    Returns:
        losses: List of loss values over training steps.
    """
    losses = []

    for idx_step in range(num_steps):
        # sample data
        X, label = dataset.sample()

        # transform data X to latent space Z
        z, logdet = model.inverse(X)

        # calculate the negative log-likelihood of X
        loss = torch.log(z.new_tensor([2 * math.pi])) + torch.mean(
            torch.sum(0.5 * z ** 2, -1) - logdet
        )

        # backpropagation and optimisation step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # log progress and record loss
        losses.append(loss.item())
        if (idx_step + 1) % 1000 == 0:
            logger.info(f"Step {idx_step + 1}/{num_steps}, Loss: {loss.item():.5f}")

    return losses
