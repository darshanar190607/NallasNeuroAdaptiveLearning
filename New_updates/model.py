import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BiLSTM Model Class
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.fc1(out[:, -1, :]))
        out = F.relu(self.fc2(out))
        out = self.fc3(self.dropout(out))

        return out

# Load the model
input_size = 14
hidden_size = 256
num_layers = 2
num_classes = 3

model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load('bilstm_model.pth', map_location=device))
model.eval()

# Load the scaler
scaler = joblib.load('scaler.pkl')

def preprocess_eeg_data(eeg_data):
    """
    Preprocess EEG data: scale and reshape for model input.
    eeg_data: list of floats, length 3584 (256 samples * 14 channels)
    """
    eeg_array = np.array(eeg_data).reshape(256, 14)  # Reshape to (256, 14)
    eeg_scaled = scaler.transform(eeg_array)  # Scale
    eeg_tensor = torch.tensor(eeg_scaled, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dim
    return eeg_tensor

def predict_attention_state(eeg_data):
    """
    Predict attention state from EEG data.
    Returns: 'Focused', 'Unfocused', or 'Drowsy'
    """
    input_tensor = preprocess_eeg_data(eeg_data)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = predicted.item()
        if pred_class == 0:
            return 'Focused'
        elif pred_class == 1:
            return 'Unfocused'
        else:
            return 'Drowsy'
