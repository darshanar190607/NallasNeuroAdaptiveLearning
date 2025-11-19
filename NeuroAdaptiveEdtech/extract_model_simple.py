import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, num_classes, dropout):
        super(CNN_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.dropout(self.fc1(out[:, -1, :])))
        out = self.fc2(self.dropout(out))

        return out

print("Creating model...")
model = CNN_BiLSTM(14, 256, 256, 2, 3, 0.5)
torch.save(model.state_dict(), 'cnn_bilstm_model.pth')
print("Model saved as cnn_bilstm_model.pth")

print("Creating scaler...")
scaler = StandardScaler()
dummy_data = np.random.normal(0, 50, (1000, 14))
scaler.fit(dummy_data)
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")

print("Setup complete!")