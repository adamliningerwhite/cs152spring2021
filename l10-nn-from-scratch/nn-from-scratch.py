#!/usr/bin/env python3

import torch
from torch import Tensor
from typing import Tuple


def initialize_parameters(
    n0: int, n1: int, n2: int, scale: float
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Initialize parameters for a 3-layer neural network.

    Args:
        n0 (int): Number of input features (aka nx)
        n1 (int): Number of neurons in layer 1
        n2 (int): Number of output neurons
        scale (float): Scaling factor for parameters

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: weights and biases for 2 layers
    """
    # TODO: implement this function
    return W1, b1, W2, b2


def forward_propagation(
    A0: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor
) -> Tuple[Tensor, Tensor]:
    """Compute the output of a 3-layer neural network.

    Args:
        A0 (Tensor): (n0, m) input matrix (aka X)
        W1 (Tensor): (n1, n0) weight matrix
        b1 (Tensor): (n1, 1) bias matrix)
        W2 (Tensor): (n2, n1) weight matrix)
        b2 (Tensor): (n2, 1) bias matrix

    Returns:
        Tuple[Tensor, Tensor]: activations/outputs for layers 1 and 2
    """
    # TODO: implement this function
    return A1, A2.T


def get_predictions_sigmoid(
    A0: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor
) -> Tensor:
    """Convert the output of a sigmoid to zeros and ones.

    Args:
        A0 (Tensor): (n0, m) input matrix (aka X)
        W1 (Tensor): (n1, n0) weight matrix
        b1 (Tensor): (n1, 1) bias matrix)
        W2 (Tensor): (n2, n1) weight matrix)
        b2 (Tensor): (n2, 1) bias matrix

    Returns:
        Tensor: binary predictions of a 3-layer neural network
    """
    # TODO: implement this function
    return A2.round()


def backward_propagation(
    A0: Tensor, A1: Tensor, A2: Tensor, Y: Tensor, W2: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute gradients of a 3-layer neural network's parameters.

    Args:
        A0 (Tensor): (n0, m) input matrix (aka X)
        A1 (Tensor): (n1, m) output of layer 1 from forward propagation
        A2 (Tensor): (n2, m) output of layer 2 from forward propagation
        Y (Tensor): (m, 1) correct targets (aka labels)
        W2 (Tensor): (n2, n1) weight matrix)

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: gradients for weights and biases
    """
    # TODO: implement this function
    return dW1, db1, dW2, db2


def update_parameters(
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    dW1: Tensor,
    db1: Tensor,
    dW2: Tensor,
    db2: Tensor,
    lr: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Update parameters of a 3-layer neural network.

    Args:
        W1 (Tensor): (n1, n0) weight matrix
        b1 (Tensor): (n1, 1) bias matrix)
        W2 (Tensor): (n2, n1) weight matrix)
        b2 (Tensor): (n2, 1) bias matrix
        dW1 (Tensor): (n1, n0) gradient matrix
        db1 (Tensor): (n1, 1) gradient matrix)
        dW2 (Tensor): (n2, n1) gradient matrix)
        db2 (Tensor): (n2, 1) gradient matrix
        lr (float): learning rate

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: updated network parameters
    """
    # TODO: implement this function
    return W1, b1, W2, b2


def compute_cost(A2: Tensor, Y: Tensor) -> float:
    """Compute cost using binary cross entropy loss.

    Args:
        A2 (Tensor): (m, 1) matrix of neural network output values
        Y (Tensor): (m, 1) correct targets (aka labels)

    Returns:
        float: computed cost
    """
    # TODO: implement this function
    return cost


def learn(
    X: Tensor,
    Y: Tensor,
    num_hidden: int,
    param_scale: float,
    num_epochs: int,
    learning_rate: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """A function for performing batch gradient descent.

    Args:
        X (Tensor): (nx, m) matrix of input features
        Y (Tensor): (m, 1) matrix of correct targets (aka labels)
        num_hidden (int): number of neurons in layer 1
        param_scale (float): scaling factor for initializing parameters
        num_epochs (int): number of training passes through all data
        learning_rate (float): learning rate

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: parameters of a 3-layer neural network
    """
    # TODO: implement this function
    # Steps:
    # 1. initialize parameters
    # 2. loop
    #   1. compute outputs with forward propagation
    #   2. compute cost (for analysis)
    #   3. compute gradients with backward propagation
    #   4. update parameters
    # 3. return final parameters

    return W1, b1, W2, b2
