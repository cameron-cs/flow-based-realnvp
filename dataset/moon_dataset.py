from sklearn.datasets import make_moons
import torch


class MoonsDataset:
    def __init__(self, n_samples=1000, noise=0.05, device="cpu"):
        self.n_samples = n_samples
        self.noise = noise
        self.device = device

    def sample(self):
        X, labels = make_moons(n_samples=self.n_samples, noise=self.noise)
        X = torch.Tensor(X).to(self.device)
        return X, labels
