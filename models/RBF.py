import torch

def RBF(X, Y, gamma=None, squared=False):

    # Y: B * 171 * 3
    # X: B * 1 * 2000 * 3
    if gamma is None:
        gamma = 1.0 / X.shape[2]

    size = (X.shape[0], X.shape[1], Y.shape[1], Y.shape[2])

    x = X.unsqueeze(2).expand(size)
    y = Y.unsqueeze(1).expand(size)

    if squared:
        euclidean = (x - y).pow(2).sum(-1)
    else:
        euclidean = (x - y).pow(2).sum(-1).pow(0.5)

    kernel = torch.exp(-1 * gamma * euclidean)
    print(f"kernel size: {kernel.shape}")
    return kernel.permute((0, 2, 1))
