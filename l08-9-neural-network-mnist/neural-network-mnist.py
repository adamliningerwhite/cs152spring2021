#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import Tuple


# In[ ]:


def get_mnist_subset_loader(
    train: bool, path: str, c1: int, c2: int
) -> Tuple[DataLoader, int]:
    """Return an MNIST dataloader for the two specified classes.

    Args:
        train (bool): Should this be a training set or validation set
        path (str): The directory in which to store/find the MNIST dataset
        c1 (int): a number in [0, 9] denoting a MNIST class/number
        c2 (int): a number in [0, 9] denoting a MNIST class/number

    Returns:
        Tuple[DataLoader, int]: Return a dataloader and its size
    """

    # All inputs must be converted into torch tensors, and the normalization values
    # have been precomputed and provided below.
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),]
    )

    dataset = MNIST(root=path, train=train, download=True, transform=mnist_transforms)

    # Grab indices for the two classes we care about
    idx_classA = [i for i, t in enumerate(dataset.targets) if t == c1]
    idx_classB = [i for i, t in enumerate(dataset.targets) if t == c2]

    idxs = idx_classA + idx_classB
    size = len(idxs)

    loader = DataLoader(dataset, sampler=SubsetRandomSampler(idxs), batch_size=size)

    return loader, size


def get_mnist_data_binary(
    path: str, c1: int, c2: int
) -> Tuple[DataLoader, int, DataLoader, int]:
    """Return data loaders for two classes from MNIST.

    Args:
        path (str): The directory in which to store/find the MNIST dataset
        c1 (int): a number in [0, 9] denoting a MNIST class/number
        c2 (int): a number in [0, 9] denoting a MNIST class/number

    Returns:
        Tuple[DataLoader, int, DataLoader, int]: Return a training dataloader the
            training set size (and the same for the validation dataset)
    """

    train_loader, train_size = get_mnist_subset_loader(True, path, c1, c2)
    valid_loader, valid_size = get_mnist_subset_loader(False, path, c1, c2)

    return train_loader, train_size, valid_loader, valid_size


def show_image(img, ax=None, title=None):
    if not ax:
        _, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.axis("off")
    if title:
        ax.set_title(title)


# In[ ]:


path = "../data"

classA = 3
classB = 8

train_loader, train_size, valid_loader, valid_size = get_mnist_data_binary(
    path, classA, classB
)


# In[ ]:


train_imgs, train_trgs = next(iter(train_loader))
valid_imgs, valid_trgs = next(iter(valid_loader))


# In[ ]:


# Neural network architecture
n0 = 28 * 28
n1 = 10
n2 = 1

# Parameters
W1 = torch.randn(n1, n0) * 0.01
b1 = torch.zeros(n1, 1)
W2 = torch.randn(n2, n1) * 0.01
b2 = torch.zeros(n2, 1)


# In[ ]:


train_imgs.shape


# In[ ]:


W1 @ train_imgs


# In[ ]:


W1 @ train_imgs.view(-1, n0).T


# In[ ]:


# Forward pass (get the predictions)
Z1 = W1 @ train_imgs.view(-1, n0).T + b1
A1 = torch.sigmoid(Z1)
Z2 = W2 @ A1 + b2
A2 = torch.sigmoid(Z2)


# In[ ]:


Z1.shape, A1.shape, Z2.shape, A2.shape


# In[ ]:


def forward(A0, W1, b1, W2, b2):
    Z1 = W1 @ A0 + b1
    A1 = torch.sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = torch.sigmoid(Z2)
    return A2


# In[ ]:


A0 = train_imgs.view(-1, n0).T
Y = train_trgs

Yhat = forward(A0, W1, b1, W2, b2)
Yhat.shape, Y.shape


# In[ ]:


def prediction(A0, W1, b1, W2, b2):
    return forward(A0, W1, b1, W2, b2).round()


# In[ ]:


preds = prediction(A0, W1, b1, W2, b2)
preds.shape


# In[ ]:


preds


# In[ ]:


train_trgs


# In[ ]:


Y = torch.zeros_like(train_trgs)
Y[train_trgs == classB] = 1


# In[ ]:


Y.sum(), train_trgs.shape


# In[ ]:


train_accuracy = (Y - preds).abs().mean()
train_accuracy


# $$
# \begin{align}
# dZ^{[2]} &= A^{[2]} - Y\\
# dW^{[2]} &= \frac{1}{m} dZ^{[2]} A^{[1]T}\\
# db^{[2]} &= \frac{1}{m} \sum dZ^{[2]}\\
# dZ^{[1]} &= W^{[2]T} dZ^{[2]} A^{[1]} (1 - A^{[1]})\\
# dW^{[1]} &= \frac{1}{m} dZ^{[1]} A^{[0]T}\\
# db^{[1]} &= \frac{1}{m} \sum dZ^{[1]}\\
# \end{align}
# $$

# In[ ]:


def forward(A0, W1, b1, W2, b2, return_state=False):
    Z1 = W1 @ A0 + b1
    A1 = torch.sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = torch.sigmoid(Z2)
    
    if return_state:
        return A1, A2
    else:
        return A2

def backward(W2, A0, A1, A2, Y):
    """With sigmoid activations and cross entropy loss."""
    
    m = len(Y)
    
    dZ2 = (A2 - Y)
    dW2 = (1/m) * (dZ2 @ A1.T)
    db2 = (1/m) * dZ2.sum(axis=1, keepdims=True)

    dZ1 = W2.T @ dZ2 * (A1 * (1 - A1))
    dW1 = (1/m) * dZ1 @ A0.T
    db1 = (1/m) * dZ1.sum(axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2    


# # Complete NN Training on MNIST
# 
# 1. Initialize parameters
# 2. Train parameters
#     1. Compute predictions
#     2. Compute gradients
#     3. Update parameters

# In[ ]:


# MNIST data
path = "../data"

classA = 3
classB = 8

train_loader, train_size, valid_loader, valid_size = get_mnist_data_binary(
    path, classA, classB
)

A0 = train_imgs.view(-1, 28*28).T
Y = torch.zeros_like(train_trgs.view(1, -1))
Y[train_trgs.view(1, -1) == classB] = 1

A0_validation = valid_imgs.view(-1, n0).T
Y_validation = torch.zeros_like(valid_trgs.view(1, -1))
Y_validation[valid_trgs.view(1, -1) == classB] = 1

# Optimization hyperparameters
learning_rate = 0.0001
num_epochs = 8

# Neural network architecture and parameters
n0 = A0.shape[0]
n1 = 10
n2 = 1

W1 = torch.randn(n1, n0) * 0.01
b1 = torch.zeros(n1, 1)
W2 = torch.randn(n2, n1) * 0.01
b2 = torch.zeros(n2, 1)

# Compute initial accuracy
valid_preds = prediction(A0_validation, W1, b1, W2, b2)
valid_accuracy = 1 - (Y_validation - valid_preds).abs().mean()
print(valid_accuracy.item())

for _ in range(num_epochs):
    A1, A2 = forward(A0, W1, b1, W2, b2, return_state=True)
    
    dW1, db1, dW2, db2 = backward(W2, A0, A1, A2, Y)
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
        
    valid_preds = prediction(A0_validation, W1, b1, W2, b2)
    valid_accuracy = 1 - (Y_validation - valid_preds).abs().mean()
    print(valid_accuracy.item())


# In[ ]:




