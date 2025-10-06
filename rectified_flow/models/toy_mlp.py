import torch
import torch.nn as nn


# Generic MLP model
class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU, linear_layer=nn.Linear):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.linear_layer = linear_layer
        for i in range(len(layer_sizes) - 1):
            self.layers.append(linear_layer(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(activation())

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x


# Model for velocity field v(x, t)
class MLPVelocity(nn.Module):
    def __init__(self, dim, hidden_sizes=[128, 128, 128], output_dim=None):
        super().__init__()
        output_dim = output_dim or dim
        self.mlp = MLP([dim + 1, *hidden_sizes, output_dim])

    def forward(self, x, t):
        t = t.squeeze().view(t.shape[0], -1)
        x = x.view(x.shape[0], -1)
        return self.mlp(torch.cat((x, t), dim=1))


# Model for velocity field v(x,t) with label conditioning
class MLPVelocityConditioned(nn.Module):
    def __init__(self, dim, hidden_sizes=[128, 128, 128], output_dim=None, label_dim=1):
        super().__init__()
        output_dim = output_dim or dim
        self.label_dim = label_dim
        self.mlp = MLP([dim + 1 + label_dim, *hidden_sizes, output_dim])

    def forward(self, x, t, labels=None):
        t = t.squeeze().view(t.shape[0], -1)
        x = x.view(x.shape[0], -1)
        if labels is not None:
            labels = labels.float().view(labels.shape[0], -1)
            x_t = torch.cat((x, t, labels), dim=1)
        else:
            x_t = torch.cat((x, t), dim=1)
        return self.mlp(x_t)
