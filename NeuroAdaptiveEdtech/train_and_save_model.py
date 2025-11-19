import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from torch.utils.data import Dataset, DataLoader

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

# Dataset class
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_synthetic_data():
    """Create synthetic EEG data for training"""
    print("Creating synthetic EEG data...")
    
    # Parameters
    n_samples = 5000
    sequence_length = 256
    n_channels = 14
    n_classes = 3
    
    data = []
    labels = []
    
    for i in range(n_samples):
        # Generate class label
        class_label = i % n_classes
        
        # Generate synthetic EEG data based on class
        sample = np.zeros((sequence_length, n_channels))
        
        for t in range(sequence_length):
            for ch in range(n_channels):
                # Base signal
                base_freq = 10 + class_label * 5  # Different frequencies for different states
                signal = np.sin(2 * np.pi * base_freq * t / sequence_length)
                
                # Add noise
                noise = np.random.normal(0, 0.3)
                
                # Add class-specific patterns
                if class_label == 0:  # Focused
                    signal += np.sin(2 * np.pi * 20 * t / sequence_length) * 0.5
                elif class_label == 1:  # Unfocused
                    signal += np.random.normal(0, 0.5)
                else:  # Drowsy
                    signal += np.sin(2 * np.pi * 5 * t / sequence_length) * 0.8
                
                sample[t, ch] = signal + noise
        
        data.append(sample)
        labels.append(class_label)
    
    return np.array(data), np.array(labels)

def train_model():
    """Train the CNN-BiLSTM model"""
    print("Training CNN-BiLSTM model...")
    
    # Create synthetic data
    X, y = create_synthetic_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 14)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 14)).reshape(X_test.shape)
    
    # Create datasets
    train_dataset = EEGDataset(X_train_scaled, y_train)
    test_dataset = EEGDataset(X_test_scaled, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model parameters
    input_size = 14
    hidden_size = 256
    sequence_length = 256
    num_layers = 2
    num_classes = 3
    dropout = 0.5
    
    # Initialize model
    model = CNN_BiLSTM(input_size, hidden_size, sequence_length, num_layers, num_classes, dropout)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluate model
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.numpy())
            actuals.extend(target.numpy())
    
    accuracy = accuracy_score(actuals, predictions)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Save model and scaler
    torch.save(model.state_dict(), 'cnn_bilstm_model.pth')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Model and scaler saved successfully!")
    return model, scaler

if __name__ == "__main__":
    print("Training BCI Model for NeuroAdaptive EdTech")
    print("=" * 50)
    
    # Train and save model
    model, scaler = train_model()
    
    print("\nFiles created:")
    print("- cnn_bilstm_model.pth (Trained PyTorch model)")
    print("- scaler.pkl (StandardScaler for preprocessing)")
    print("\nYou can now run the FastAPI server with:")
    print("python fastapi_server.py")