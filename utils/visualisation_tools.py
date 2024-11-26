import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_moons


def plot_forward_transformation(X, label, z, title1="X sampled from Moon dataset", title2="Z transformed from X"):
    """
    Plot the data before and after forward transformation.
    """
    X = X.cpu().detach().numpy()
    z = z.cpu().detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].scatter(X[label == 0, 0], X[label == 0, 1], label="Class 0")
    axes[0].scatter(X[label == 1, 0], X[label == 1, 1], label="Class 1")
    axes[0].set_title(title1)
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].legend()

    axes[1].scatter(z[label == 0, 0], z[label == 0, 1], label="Class 0")
    axes[1].scatter(z[label == 1, 0], z[label == 1, 1], label="Class 1")
    axes[1].set_title(title2)
    axes[1].set_xlabel(r"$z_1$")
    axes[1].set_ylabel(r"$z_2$")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_reverse_transformation(z, X_transformed, title1="Z sampled from normal distribution",
                                title2="X transformed from Z"):
    """
    Plot the data before and after reverse transformation.
    """
    z = z.cpu().detach().numpy()
    X_transformed = X_transformed.cpu().detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].scatter(z[:, 0], z[:, 1])
    axes[0].set_title(title1)
    axes[0].set_xlabel(r"$z_1$")
    axes[0].set_ylabel(r"$z_2$")

    axes[1].scatter(X_transformed[:, 0], X_transformed[:, 1])
    axes[1].set_title(title2)
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")

    plt.tight_layout()
    plt.show()


def plot_loss_curve(losses):
    """
    Plot the training loss curve.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, label="Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()


def animate_transformation(X, realNVP, label):
    """
    Create an animation showing the evolution of the data across affine coupling layers.
    """
    frames = []
    y = X
    for affine_coupling in realNVP.affine_couplings:
        y, _ = affine_coupling(y)
        frames.append(y.cpu().detach().numpy())

    def update(frame):
        plt.clf()
        plt.scatter(frame[:, 0], frame[:, 1], c=label, cmap="viridis", s=10)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.title("Transforming Data Across Layers")

    fig = plt.figure(figsize=(6, 6))
    anim = FuncAnimation(fig, update, frames=frames, interval=500)
    plt.show()


def plot_reconstruction(X, z, X_reconstructed, label):
    """
    Plot original data, latent space, and reconstructed data in a single figure.
    """
    X = X.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    X_reconstructed = X_reconstructed.cpu().detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].scatter(X[:, 0], X[:, 1], c=label, cmap="viridis", s=10)
    axes[0].set_title("Original Data (X)")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")

    axes[1].scatter(z[:, 0], z[:, 1], c=label, cmap="viridis", s=10)
    axes[1].set_title("Latent Space (Z)")
    axes[1].set_xlabel(r"$z_1$")
    axes[1].set_ylabel(r"$z_2$")

    axes[2].scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], c=label, cmap="viridis", s=10)
    axes[2].set_title("Reconstructed Data (X')")
    axes[2].set_xlabel(r"$x_1$")
    axes[2].set_ylabel(r"$x_2$")

    plt.tight_layout()
    plt.show()


# visualisation process
def visualise_training(realNVP, losses, device, n_samples=1000, noise=0.05):
    """
    Generate all visualsations after training.
    """
    # forward transformation visualisation
    X, label = make_moons(n_samples=n_samples, noise=noise)
    X = torch.Tensor(X).to(device=device)
    z, logdet_jacobian = realNVP.inverse(X)
    plot_forward_transformation(X, label, z)

    # reverse transformation visualisation
    z = torch.normal(0, 1, size=(1000, 2)).to(device=device)
    X_transformed, _ = realNVP(z)
    plot_reverse_transformation(z, X_transformed)

    # loss curve
    plot_loss_curve(losses)

    # animation of data transformation
    animate_transformation(X, realNVP, label)

    # reconstruction visualisation
    with torch.no_grad():
        z, _ = realNVP.inverse(X)
        X_reconstructed, _ = realNVP(z)
    plot_reconstruction(X, z, X_reconstructed, label)
