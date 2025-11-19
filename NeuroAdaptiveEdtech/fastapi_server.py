from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List, Dict
import uvicorn
import sys
import os

# Add New_updates directory to path for model imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'New_updates'))

try:
    from model import predict_attention_state as new_predict_function
    NEW_MODEL_AVAILABLE = True
    print("✅ New BCI model loaded from New_updates folder")
except ImportError as e:
    NEW_MODEL_AVAILABLE = False
    print(f"⚠️ Could not load new model: {e}")
    print("📦 Using fallback CNN-BiLSTM model")

app = FastAPI(title="NeuroAdaptive BCI API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class EEGData(BaseModel):
    eeg_data: List[List[float]]

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

# CNN-BiLSTM Model Definition
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

# Global variables
model = None
scaler = None
classes = ['Focused', 'Unfocused', 'Drowsy']

# Model parameters
input_size = 14
hidden_size = 256
sequence_length = 256
num_layers = 2
num_classes = 3
dropout = 0.5

def load_model():
    global model, scaler
    try:
        # Initialize model
        model = CNN_BiLSTM(input_size, hidden_size, sequence_length, num_layers, num_classes, dropout)
        
        # Try to load trained model weights
        try:
            model.load_state_dict(torch.load('cnn_bilstm_model.pth', map_location=torch.device('cpu')))
            print("✅ Trained model loaded successfully")
        except FileNotFoundError:
            print("⚠️ No trained model found, using random weights")
        
        model.eval()

        # Try to load scaler
        try:
            scaler = joblib.load('scaler.pkl')
            print("✅ Scaler loaded successfully")
        except FileNotFoundError:
            print("⚠️ No scaler found, creating default StandardScaler")
            scaler = StandardScaler()
            # Fit with dummy data for consistent behavior
            dummy_data = np.random.randn(1000, 14)
            scaler.fit(dummy_data)

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {"message": "NeuroAdaptive BCI API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_attention_state(data: EEGData):
    try:
        # Try using the new model first
        if NEW_MODEL_AVAILABLE:
            try:
                eeg_data = np.array(data.eeg_data)
                
                # Flatten for new model (expects 3584 values)
                if eeg_data.shape == (sequence_length, 14):
                    eeg_flat = eeg_data.flatten().tolist()
                else:
                    eeg_flat = eeg_data.tolist() if len(eeg_data.shape) == 1 else eeg_data.flatten().tolist()
                
                if len(eeg_flat) != 3584:
                    # Pad or truncate to expected size
                    if len(eeg_flat) < 3584:
                        eeg_flat.extend([0.0] * (3584 - len(eeg_flat)))
                    else:
                        eeg_flat = eeg_flat[:3584]
                
                prediction = new_predict_function(eeg_flat)
                
                # Convert to expected format
                if prediction == 'Focused':
                    probs = {'Focused': 0.8, 'Unfocused': 0.15, 'Drowsy': 0.05}
                elif prediction == 'Unfocused':
                    probs = {'Focused': 0.2, 'Unfocused': 0.7, 'Drowsy': 0.1}
                else:  # Drowsy
                    probs = {'Focused': 0.1, 'Unfocused': 0.2, 'Drowsy': 0.7}
                
                return PredictionResponse(
                    prediction=prediction,
                    confidence=probs[prediction],
                    probabilities=probs
                )
                
            except Exception as new_model_error:
                print(f"⚠️ New model failed: {new_model_error}")
                print("🔄 Falling back to CNN-BiLSTM model")
        
        # Fallback to original model
        if model is None:
            raise HTTPException(status_code=500, detail="No model available")

        eeg_data = np.array(data.eeg_data)
        
        # Validate input shape
        if eeg_data.shape != (sequence_length, 14):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid data shape. Expected ({sequence_length}, 14), got {eeg_data.shape}"
            )

        # Scale the data
        if scaler:
            eeg_data = scaler.transform(eeg_data)

        # Convert to tensor
        eeg_tensor = torch.FloatTensor(eeg_data).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(eeg_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = F.softmax(outputs, dim=1).numpy()[0]

        prediction = classes[predicted.item()]
        confidence = float(probabilities[predicted.item()])

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities={
                classes[i]: float(probabilities[i]) for i in range(len(classes))
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate")
async def simulate_prediction():
    """Generate a simulated prediction for testing"""
    try:
        # Generate random probabilities that sum to 1
        probs = np.random.dirichlet([1, 1, 1])
        predicted_idx = np.argmax(probs)
        
        return PredictionResponse(
            prediction=classes[predicted_idx],
            confidence=float(probs[predicted_idx]),
            probabilities={
                classes[i]: float(probs[i]) for i in range(len(classes))
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)