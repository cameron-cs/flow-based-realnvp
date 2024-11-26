import torch
import torch.nn as nn
import torch.nn.init as init


class MaskBlock(nn.Module):
    """
    Mask to separate positions that do not change and positions that change.
    mask[i] = 1 means the i-th position does not change.
    """

    def __init__(self, mask):
        super(MaskBlock, self).__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)

    def apply_mask(self, x):
        return x * self.mask

    def apply_inverse_mask(self, x):
        return x * (1 - self.mask)


class ScaleBlock(nn.Module):
    """
    The block to compute the scaling factor in affine transformation.
    """

    def __init__(self, input_dim, hidden_dim):
        super(ScaleBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.scale = nn.Parameter(torch.Tensor(input_dim))
        init.normal_(self.scale)

    def forward(self, x):
        s = torch.relu(self.fc1(x))
        s = torch.relu(self.fc2(s))
        s = self.fc3(s) * self.scale
        return s


class TranslationBlock(nn.Module):
    """
    The block to compute the translation in affine transformation.
    """

    def __init__(self, input_dim, hidden_dim):
        super(TranslationBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        t = torch.relu(self.fc1(x))
        t = torch.relu(self.fc2(t))
        t = self.fc3(t)
        return t
