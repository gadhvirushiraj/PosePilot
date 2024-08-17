"""
This file contains the model architecture for the classification task.
"""

import torch.nn as nn


class ClassifyPose(nn.Module):
    """
    The model architecture for the classification task.
    """

    def __init__(
        self, input_size, hidden_size, num_layers, sequence_length, num_classes
    ):
        super(ClassifyPose, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=0.4, batch_first=True
        )
        # Attention layer
        # self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

        self.fc = nn.Linear(hidden_size * sequence_length, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.leaky_relu = nn.ReLU(0.1)

    def forward(self, x):

        out, _ = self.lstm(x)
        # Apply attention
        # out, _ = self.attention(out, out, out)

        # Decode the hidden state of the last time step
        # out.shape = (batch_size, seq_length, hidden_size)
        out = out.contiguous().view(out.size(0), -1)
        out = self.fc(out)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = self.fc3(out)

        return out
