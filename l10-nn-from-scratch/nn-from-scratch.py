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
    W1 = torch.randn(n1, n0) * scale
    b1 = torch.zeros(n1, 1)
    W2 = torch.randn(n2, n1) * scale
    b2 = torch.zeros(n2, 1) 
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
        Tuple[Tensor, Tensor]: outputs for layers 1 (n1, m) and 2 (n2, m)
    """
    # Step 1: Compute output of first layer 
    Z1 = W1 @ A0 + b1 
    A1 = torch.sigmoid(Z1)

    # Step 2: Compute output of second layer 
    Z2 = W2 @ A1 + b2
    A2 = torch.sigmoid(Z2)

    # Return outputs of both layers 
    return A1, A2

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
    # Get neural network output 
    A1, A2 = forward_propagation(A0, W1, b1, W2, b2)
    # Round to get predictions
    return A2.round()


def backward_propagation(
    A0: Tensor, A1: Tensor, A2: Tensor, Y: Tensor, W2: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute gradients of a 3-layer neural network's parameters.

    Args:
        A0 (Tensor): (n0, m) input matrix (aka X)
        A1 (Tensor): (n1, m) output of layer 1 from forward propagation
        A2 (Tensor): (n2, m) output of layer 2 from forward propagation
        Y (Tensor): (n2, m) correct targets (aka labels)
        W2 (Tensor): (n2, n1) weight matrix)

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: gradients for weights and biases
    """
    m = len(Y)
    # Compute partial derivatives, reusing previously computed terms as you go
    dZ2 = (A2 - Y) 
    dW2 = (1/m) * (dZ2 @ A1.T) 
    db2 = (1/m) * dZ2.sum(axis=1, keepdims=True) # Not sure why keepdims (took this from notes)
    dZ1 = W2.T @ dZ2 * (A1 * (1 - A1))
    dW1 = (1/m) * dZ1 @ A0.T
    db1 = (1/m) * dZ1.sum(axis=1, keepdims=True) # Not sure why keepdims (took this from notes)
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
    # Update layer 1 parameters 
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    # Update layer 2 parameters
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    return W1, b1, W2, b2


def compute_cost(A2: Tensor, Y: Tensor) -> Tensor:
    """Compute cost using binary cross entropy loss.

    Args:
        A2 (Tensor): (n2, m) matrix of neural network output values
        Y (Tensor): (n2, m) correct targets (aka labels)

    Returns:
        float: computed cost
    """
    m = Y.shape[1]
    losses = -(Y * torch.log(A2) + (1 - Y) * torch.log(1 - A2))
    cost = (1 / m) * losses.sum(dim=1, keepdims=True) # Not sure why keepdims (took this from notes)
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
        Y (Tensor): (n2, m) matrix of correct targets (aka labels)
        num_hidden (int): number of neurons in layer 1
        param_scale (float): scaling factor for initializing parameters
        num_epochs (int): number of training passes through all data
        learning_rate (float): learning rate

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: parameters of a 3-layer neural network
    """
    # Steps:
    # 1. initialize parameters
    n0 = X.shape[0]
    n1 = num_hidden
    n2 = Y.shape[0]
    W1, b1, W2, b2 = initialize_parameters(n0, n1, n2, param_scale)

    # 2. loop
    for epoch in range(num_epochs):
        #   1. compute outputs with forward propagation
        A1, A2 = forward_propagation(X, W1, b1, W2, b2)

        #   2. compute cost (for analysis)
        cost = compute_cost(A2, Y)

        #   3. compute gradients with backward propagation
        dW1, db1, dW2, db2 = backward_propagation(X, A1, A2, Y, W2)

        #   4. update parameters
        W1, b1, W2, b2 = update_parameters(
            W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate
        )
        
    # 3. return final parameters
    return W1, b1, W2, b2
