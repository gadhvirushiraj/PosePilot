
import torch.nn as nn

# Define LSTM model class
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)


        self.fc = nn.Linear(self.hidden_size*2, 8192)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(8192, 4096)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 512)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # # Initialize hidden state and cell state
        # h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x)

        # Decode the hidden state of the last time step :: out.shape = (batch_size, seq_length, hidden_size)
        out = out.contiguous().view(-1, self.hidden_size*2)
        # print(out.shape)

        out = self.fc(out)
        out = self.relu1(out)
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.fc2(out)
        out = self.relu3(out)
        out = self.fc3(out)
        
        return out