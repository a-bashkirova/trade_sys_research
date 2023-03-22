from torch.nn import LSTM, Linear, ReLU, Sigmoid, Dropout, Conv1d, MaxPool1d
import torch.nn as nn

class FCModel(nn.Module):

    def __init__(self, input_size):
        super(FCModel, self).__init__()
        self.fc1 = Linear(input_size, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 32)
        self.fc4 = Linear(32, 16)
        self.fc5 = Linear(16, 1)
        self.relu = ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

    def infer(self, x, threshold=.5):
        preds = Sigmoid()(self.forward(x))
        return (preds >= threshold).float()

class LSTMModel(nn.Module):

    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = LSTM(input_size=input_size, num_layers=1,
                         hidden_size=16, batch_first=True)
        self.fc1 = Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out)
        return out

    def infer(self, x, threshold=.5):
        preds = Sigmoid()(self.forward(x))
        return (preds >= threshold).float()

class CNNModel(nn.Module):

    def __init__(self, input_size):
        self.input_size = input_size
        super(CNNModel, self).__init__()
        self.conv1 = Conv1d(in_channels=1, out_channels=1, kernel_size=30)
        self.conv2 = Conv1d(in_channels=1, out_channels=1, kernel_size=7)
        self.mp = MaxPool1d(kernel_size=2, stride=1)
        self.relu = ReLU()
        self.fc1 = Linear(self.input_size - 36, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = self.relu(x)
        x = x.view(-1, self.input_size - 36)
        x = self.fc1(x)
        return x

    def infer(self, x, threshold=.5):
        preds = Sigmoid()(self.forward(x))
        return (preds >= threshold).float()