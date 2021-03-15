#!/usr/bin/env python3


import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from argparse import ArgumentParser
from time import time
from typing import Tuple, List


def get_mnist_loader(path: str, train: bool) -> Tuple[Tensor, Tensor]:
    """Return an MNIST dataloader for all ten digits.

    Args:
        path (str): Path to store/find the MNIST dataset
        train (bool): Load the training set if True, validation set if false

    Returns:
        Tuple[Tensor, Tensor]: Return images and labels
    """

    # All inputs must be converted into torch tensors, and the normalization values
    # have been precomputed and provided below.
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),]
    )

    # We'll use dataloader more later on, so I want you to get used to seeing them
    dataset = MNIST(root=path, train=train, download=True, transform=mnist_transforms)
    loader = DataLoader(dataset, batch_size=len(dataset))

    # Grab all images and targets from the loader
    images, targets = next(iter(loader))

    # Reshape the images into row vectors (instead of 28 by 28 matrices)
    m = images.shape[0]
    images = images.view(m, -1)

    return images, targets


def get_mnist_data(path: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return training and validation dataset images and labels.

    Args:
        path (str): Path to store/find the MNIST dataset

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Training images and labels, then validation
    """
    train_imgs, train_trgs = get_mnist_loader(path, train=True)
    valid_imgs, valid_trgs = get_mnist_loader(path, train=False)

    return train_imgs, train_trgs, valid_imgs, valid_trgs


def plot_learning(
    train_costs: List[float], valid_costs: List[float], valid_accuracies: List[float]
):
    """Plot learning process.

    Args:
        train_costs (List[float]): List of training costs
        valid_costs (List[float]): List of validation costs
        valid_accuracies (List[float]): List of validation accuracies
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    epochs = range(len(train_costs))

    fig.suptitle("MNIST Training")

    axes[0].plot(epochs, train_costs)
    axes[0].plot(epochs, valid_costs)
    axes[0].legend(("Training", "Validation"))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cost")

    axes[1].plot(epochs, valid_accuracies)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim((0, 1))
    axes[1].grid()

    plt.show()


def learn_nn_mnist(
    mnist_dir: str, num_epochs: int, learning_rate: float, show_plot: bool
):
    """Train a neural network on the MNIST dataset.

    Args:
        mnist_dir (str): Path to store/find the MNIST dataset
        num_epochs (int): Total number of epochs to train
        learning_rate (float): Hyperparameter controlling learning speed
        show_plot (bool): Plot learning process
    """

    train_imgs, train_trgs, valid_imgs, valid_trgs = get_mnist_data(mnist_dir)

    nx = train_imgs.shape[1]
    ny = train_trgs.unique().shape[0]

    # TODO: (DO THIS LAST)try adding additional layers, but make certain
    # that the final layer is a Linear layer with out_features=ny.
    model = torch.nn.Sequential(torch.nn.Linear(in_features=nx, out_features=ny),)

    # TODO: Create a CrossEntropyLoss function by looking at the documentation here:
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    train_costs = []
    valid_costs = []
    valid_accus = []

    for epoch in range(num_epochs):

        epoch_start = time()

        # Put the model into training mode
        model.train()

        # Forward (compute the neural network output)
        # TODO: compute the outputs of the neural network model
        train_yhat = model(train_imgs)

        # Compute cost (average loss over all examples)
        train_cost = cross_entropy_loss(train_yhat, train_trgs)

        # Compute accuracy on validation data
        model.eval()
        with torch.no_grad():
            valid_yhat = model(valid_imgs)
            valid_cost = cross_entropy_loss(valid_yhat, valid_trgs)
            predictions = valid_yhat.argmax(dim=1, keepdim=True)
            valid_accuracy = predictions.eq(valid_trgs.view_as(predictions))

            # Convert correct/incorrect matrix into a percentage
            valid_accuracy = valid_accuracy.double().mean().item()

        # Create message to print
        num_digits = len(str(num_epochs))
        msg = f"{epoch:>{num_digits}}/{num_epochs}"
        msg += f" -> T Cost: {train_cost:.3f}"
        msg += f", V Cost: {valid_cost:.3f}"
        msg += f", V Accuracy: {valid_accuracy:.3f}"

        # Put the model into training mode
        model.train()

        # Backward (compute gradients)
        # TODO: In two steps, zero out the model gradients and compute new gradients
        model.zero_grad()
        train_cost.backward()

        # Update parameters
        with torch.no_grad():
            for param in model.parameters():
                # TODO: update the model parameters
                param -= learning_rate * param.grad

        print(msg, f"  ({time() - epoch_start:.3f}s)")

        train_costs.append(train_cost)
        valid_costs.append(valid_cost)
        valid_accus.append(valid_accuracy)

    if show_plot:
        plot_learning(train_costs, valid_costs, valid_accus)


def main():

    aparser = ArgumentParser("Train a neural network on the MNIST dataset.")
    aparser.add_argument("mnist", type=str, help="Path to store/find the MNIST dataset")
    aparser.add_argument("--num_epochs", type=int, default=10)
    aparser.add_argument("--learning_rate", type=float, default=0.1)
    aparser.add_argument("--show_plot", action="store_true")
    args = aparser.parse_args()

    learn_nn_mnist(args.mnist, args.num_epochs, args.learning_rate, args.show_plot)


if __name__ == "__main__":
    main()
