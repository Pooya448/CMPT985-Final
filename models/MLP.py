import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """docstring for MLP."""

    def __init__(self, opt, channels = []):
        super(MLP, self).__init__()
        self.filters = nn.ModuleList()
        self.no_res = opt['no_residual']
        self.channels = channels
        self.lactive = opt['last_activation']
        self.nlactive = opt['nlast_activation']
        self.norm = opt['norm']


        self.last_activation = None
        if self.lactive == 'sigmoid':
            self.last_activation = nn.Sigmoid()
        elif self.lactive == 'leakyrelu':
            self.last_activation = nn.LeakyReLU()
        elif self.lactive == 'relu':
            self.last_activation = nn.ReLU()
        elif self.lactive == 'tanh':
            self.last_activation = nn.Tanh()
        elif self.lactive == 'softmax':
            self.last_activation = nn.Softmax(dim=1)
        else:
            self.last_activation = None


        self.nlast_activation = None
        if self.nlactive == 'sigmoid':
            self.nlast_activation = nn.Sigmoid()
        elif self.nlactive == 'leakyrelu':
            self.nlast_activation = nn.LeakyReLU()
        elif self.nlactive == 'relu':
            self.nlast_activation = nn.ReLU()
        elif self.nlactive == 'tanh':
            self.nlast_activation = nn.Tanh()
        elif self.nlactive == 'softmax':
            self.nlast_activation = nn.Softmax(dim=1)


        if self.no_res:
            for l in range(len(channels) - 1):
                if self.norm == 'weight' and l != len(self.channels) - 2:
                    self.filters.append(nn.utils.weight_norm(
                    nn.Conv1d(
                        self.channels[l],
                        self.channels[l + 1],
                        1)))
                else:
                    self.filters.append(
                    nn.Conv1d(
                        self.channels[l],
                        self.channels[l + 1],
                        1))
        else:
            for l in range(len(self.channels) - 1):
                if self.norm == 'weight' and l != len(self.channels) - 2:
                    self.filters.append(nn.utils.weight_norm(
                    nn.Conv1d(
                        self.channels[l] + self.channels[0],
                        self.channels[l + 1],
                        1)))
                else:
                    self.filters.append(
                    nn.Conv1d(
                        self.channels[l] + self.channels[0],
                        self.channels[l + 1],
                        1))

    def forward(self, features):

        X = features
        X0 = features

        for idx, filter in enumerate(self.filters):

            if self.no_res:
                X = filter(X)
            else:
                X = filter(torch.cat([X, X0], 1))

            if idx != len(self.channels) - 1 and self.nlast_activation != None:
                X = self.nlast_activation(X)

        if self.last_activation != None:
            X = self.last_activation(X)

        return X
