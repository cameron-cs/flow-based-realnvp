import torch
import torch.nn as nn

from model.blocks import MaskBlock
from model.layers.affine_coupling_layer import AffineCouplingLayer


class RealNVP2D(nn.Module):
    """
    A vanilla RealNVP class for modeling 2-dimensional distributions.
    """

    def __init__(self, masks, hidden_dim):
        """
        Initialise RealNVP with a list of masks. Each mask defines an affine coupling layer.
        """
        super(RealNVP2D, self).__init__()
        self.hidden_dim = hidden_dim

        # Convert masks into Mask modules
        self.masks = nn.ModuleList([MaskBlock(torch.tensor(m, dtype=torch.float32)) for m in masks])

        # Create affine coupling layers
        self.affine_couplings = nn.ModuleList([
            AffineCouplingLayer(self.masks[i].mask, self.hidden_dim) for i in range(len(self.masks))
        ])

    def forward(self, x):
        """
        Convert latent space variables into observed variables.
        """
        y = x
        logdet_tot = 0

        # Apply forward pass through all affine coupling layers
        for affine_coupling in self.affine_couplings:
            y, logdet = affine_coupling(y)
            logdet_tot += logdet

        # Apply normalisation layer (tanh-based transformation)
        normalisation_logdet = torch.sum(
            torch.log(torch.abs(4 * (1 - torch.tanh(y) ** 2))), dim=-1
        )
        y = 4 * torch.tanh(y)
        logdet_tot += normalisation_logdet

        return y, logdet_tot

    def inverse(self, y):
        """
        Convert observed variables into latent space variables.
        """
        x = y
        logdet_tot = 0

        # Inverse the normalisation layer
        normalisation_logdet = torch.sum(
            torch.log(torch.abs(1.0 / 4.0 * 1 / (1 - (x / 4) ** 2))), dim=-1
        )
        x = 0.5 * torch.log((1 + x / 4) / (1 - x / 4))
        logdet_tot += normalisation_logdet

        # Inverse affine coupling layers
        for affine_coupling in reversed(self.affine_couplings):
            x, logdet = affine_coupling.inverse(x)
            logdet_tot += logdet

        return x, logdet_tot