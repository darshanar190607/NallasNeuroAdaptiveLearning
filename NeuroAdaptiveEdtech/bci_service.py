from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the scaler (we'll need to save it from the notebook)
scaler = None

# Define the CNN-BiLSTM model (same as in notebook)
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, num_classes, dropout):
        super(CNN_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN layers
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM layers
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Fully connected layers
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

# Model parameters
input_size = 14
hidden_size = 256
sequence_length = 256
num_layers = 2
num_classes = 3
dropout = 0.5

# Initialize model
model = CNN_BiLSTM(input_size, hidden_size, sequence_length, num_layers, num_classes, dropout)
model.eval()

# Class labels
classes = ['Focused', 'Unfocused', 'Drowsy']

def load_model():
    global scaler
    try:
        # Load model state
        model.load_state_dict(torch.load('cnn_bilstm_model.pth', map_location=torch.device('cpu')))
        print("Model loaded successfully")

        # Load scaler (we'll create this)
        try:
            scaler = joblib.load('scaler.pkl')
            print("Scaler loaded successfully")
        except:
            print("Scaler not found, using default")
            scaler = StandardScaler()

    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        eeg_data = np.array(data['eeg_data'])  # Expected shape: (sequence_length, 14)

        if eeg_data.shape != (sequence_length, 14):
            return jsonify({'error': f'Invalid data shape. Expected ({sequence_length}, 14), got {eeg_data.shape}'}), 400

        # Scale the data
        if scaler:
            eeg_data = scaler.transform(eeg_data)

        # Convert to tensor
        eeg_tensor = torch.FloatTensor(eeg_data).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(eeg_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = F.softmax(outputs, dim=1).numpy()[0]

        prediction = classes[predicted.item()]
        confidence = float(probabilities[predicted.item()])

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                classes[i]: float(probabilities[i]) for i in range(len(classes))
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)