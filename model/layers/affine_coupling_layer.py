import torch
import torch.nn as nn

from model.blocks import MaskBlock, ScaleBlock, TranslationBlock


class AffineCouplingLayer(nn.Module):
    """
    Affine Coupling layer for RealNVP.
    Combines scale and translation networks with masking for transformation.
    """

    def __init__(self, mask, hidden_dim):
        super(AffineCouplingLayer, self).__init__()
        self.mask = MaskBlock(mask)
        self.scale_network = ScaleBlock(len(mask), hidden_dim)
        self.translation_network = TranslationBlock(len(mask), hidden_dim)

    def forward(self, x):
        """
        Convert latent space variable to observed variable.
        """
        masked_x = self.mask.apply_mask(x)
        s = self.scale_network(masked_x)
        t = self.translation_network(masked_x)

        y = masked_x + self.mask.apply_inverse_mask(x * torch.exp(s) + t)
        logdet = torch.sum(self.mask.apply_inverse_mask(s), dim=-1)

        return y, logdet

    def inverse(self, y):
        """
        Convert observed variable to latent space variable.
        """
        masked_y = self.mask.apply_mask(y)
        s = self.scale_network(masked_y)
        t = self.translation_network(masked_y)

        x = masked_y + self.mask.apply_inverse_mask((y - t) * torch.exp(-s))
        logdet = torch.sum(self.mask.apply_inverse_mask(-s), dim=-1)

        return x, logdet