import torch
import torch.nn as nn
import torch.nn.functional as F

def ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.SiLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Dropout2d(dropout)
    )

def FeedForwardBlock(in_features, out_features, dropout=0.2):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.SiLU(inplace=True),
        nn.BatchNorm1d(out_features),
        nn.Dropout(dropout)
    )

def FeedForwardLastLayer(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
    )

class ComposableCNN(nn.Module):
    def __init__(self, conv_params):
        super(ComposableCNN, self).__init__()

        self.n_conv_layers = len(conv_params)        
        for i, (in_ch, out_ch, k, s, p, d) in enumerate(conv_params):
            setattr(self, f'block{i+1}', ConvBlock(in_ch, out_ch, k, s, p, d))

    def forward(self, x):
        for i in range(1, self.n_conv_layers + 1):
            block = getattr(self, f'block{i}')
            x = block(x)
        return x

class ComposableFFN(nn.Module):
    def __init__(self, fc_params, dropout=0.2):
        super(ComposableFFN, self).__init__()
        self.n_ff_layers = len(fc_params)        
        for i, (in_ch, out_ch, d) in enumerate(fc_params):
            if i == self.n_ff_layers - 1:
                setattr(self, f'block{i+1}', FeedForwardLastLayer(in_ch, out_ch))
            else:
                setattr(self, f'block{i+1}', FeedForwardBlock(in_ch, out_ch, d))

    def forward(self, x):
        for i in range(1, self.n_ff_layers + 1):
            block = getattr(self, f'block{i}')
            x = block(x)
        return x

class PediatricCXRClassificationModel(nn.Module):
    def __init__(self, conv_params, fc_params):
        super(PediatricCXRClassificationModel, self).__init__()
        self.cnn = ComposableCNN(conv_params)
        self.ffn = ComposableFFN(fc_params)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.ffn(x)
        return x

