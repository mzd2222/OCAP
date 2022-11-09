import torch
import numpy as np
from torch import nn

class channel_selection(nn.Module):
    """
    Auxiliary resnet channel pruning
    """
    def __init__(self, num_channels):

        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))
        self.indexes.requires_grad = False

    def forward(self, input_tensor):
        """
        input dim: (N,C,H,W), also the output dim of BN
        """
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, selected_index, :, :]
        return output
