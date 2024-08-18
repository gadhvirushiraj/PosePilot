"""
This file contains the model architecture for the Correction task.
"""

import torch.nn as nn

class RNN(nn.Module):
    """
    The model architecture for the classification task.

    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer with bidirectional support and dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.5)

        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4, batch_first=True)

        self.fc = nn.Linear(self.hidden_size*2, 8192)
        self.relu1 = nn.LeakyReLU()
        self.fc1 = nn.Linear(8192, 4096)
        self.relu2 = nn.LeakyReLU()
        self.fc2 = nn.Linear(4096, 512)
        self.relu3 = nn.LeakyReLU()
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm(x)

        # Apply attention
        out, att_weights = self.attention(out, out, out)

        # Decode the hidden state of the last time step :: out.shape = (batch_size, seq_length, hidden_size)
        out = out.contiguous().view(-1, self.hidden_size*2)

        out = self.fc(out)
        out = self.relu1(out)
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.fc2(out)
        out = self.relu3(out)
        out = self.fc3(out)
        
        return out, att_weights
