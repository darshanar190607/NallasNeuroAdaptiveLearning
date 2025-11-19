import os
import scipy.io
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Constants
FOCUSED_CLASS = 0
UNFOCUSED_CLASS = 1
DROWSY_CLASS = 2

columns = [
    'ED_COUNTER', 'ED_INTERPOLATED', 'ED_RAW_CQ', 'ED_AF3', 'ED_F7',
    'ED_F3', 'ED_FC5', 'ED_T7', 'ED_P7', 'ED_O1',
    'ED_O2', 'ED_P8', 'ED_T8', 'ED_FC6', 'ED_F4',
    'ED_F8', 'ED_AF4', 'ED_GYROX', 'ED_GYROY', 'ED_TIMESTAMP',
    'ED_ES_TIMESTAMP', 'ED_FUNC_ID', 'ED_FUNC_VALUE', 'ED_MARKER', 'ED_SYNC_SIGNAL'
]

def get_state(timestamp):
    if timestamp <= 10*128*60:
        return FOCUSED_CLASS
    elif timestamp > 20*128*60:
        return UNFOCUSED_CLASS
    else:
        return DROWSY_CLASS

# Global scaler
scaler = StandardScaler()

def get_EEG_data(data_root, filename):
    hz = 128
    mat = scipy.io.loadmat(os.path.join(data_root, filename))
    data = mat["o"]["data"][0,0]
    eeg_df = pd.DataFrame(data, columns=columns)
    eeg_df = eeg_df.filter(['ED_AF3', 'ED_F7', 'ED_F3', 'ED_FC5',
                            'ED_T7', 'ED_P7', 'ED_O1', 'ED_O2',
                            'ED_P8', 'ED_T8', 'ED_FC6', 'ED_F4',
                            'ED_F8', 'ED_AF4'])
    labels = ['AF3','F7', 'F3','FC5','T7','P7','O1','O2','P8','T8', 'FC6','F4','F8','AF4']
    eeg_df.columns = labels
    eeg_df = pd.DataFrame(scaler.fit_transform(eeg_df), columns=eeg_df.columns)
    eeg_df.reset_index(inplace=True)
    eeg_df.rename(columns={'index': 'timestamp'}, inplace=True)
    eeg_df['state'] = eeg_df['timestamp'].apply(get_state)
    return eeg_df

# Load data from local mat file
data_root = ''
files = ['eeg_record1_copy.mat']  # Assuming single file

dataset = []
for filename in files:
    data = get_EEG_data(data_root, filename)
    dataset.append(data)

def split_epochs(data, hz, epoch_length=2, step_size=0.125):
    step = int(epoch_length * hz - step_size * hz)
    offset = int(epoch_length * hz)
    starts = []
    current = 0
    while current + offset <= data.shape[0]:
        starts.append(current)
        current += step
    ends = [x + offset for x in starts]
    epochs = []
    for i in range(len(starts)):
        epoch = data.iloc[starts[i]:ends[i]]
        epochs.append(epoch)
    return epochs

epochs_data = []
for eeg in dataset:
    epochs = split_epochs(eeg, 128)
    for epoch in epochs:
        epochs_data.append(epoch)

class EEGDataset(Dataset):
    def __init__(self, dataframes, target_column='state'):
        self.data = []
        self.targets = []
        for df in dataframes:
            self.targets.append(df[target_column].mode()[0])
            feature = df.drop(columns=[target_column, 'timestamp'], errors='ignore')
            self.data.append(feature.values)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

dataset = EEGDataset(epochs_data)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

learning_rate = 0.001
num_epochs = 50
input_size = 14
num_layers = 2
hidden_size = 256
num_classes = 3

bilstm = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bilstm.parameters(), lr=learning_rate)
early_stopping = EarlyStopping()

def training(model, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_labels = []
        train_preds = []
        for data, targets in tqdm(train_loader):
            data = data.to(device=device).squeeze(1)
            targets = targets.to(device=device).long()
            detections = model(data)
            loss = criterion(detections, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(detections, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            train_labels.extend(targets.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        f1 = f1_score(train_labels, train_preds, average='macro')
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.2f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.2f}')
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device=device).squeeze(1)
                targets = targets.to(device=device).long()
                detections = model(data)
                loss = criterion(detections, targets)
                running_loss += loss.item()
                _, predicted = torch.max(detections, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                val_labels.extend(targets.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
            epoch_loss = running_loss / len(val_loader)
            accuracy = 100 * correct / total
            f1 = f1_score(val_labels, val_preds, average='macro')
            print(f'Validation Loss: {epoch_loss:.2f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.2f}')
        early_stopping(epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

training(bilstm, num_epochs)

# Load best model
early_stopping.load_best_model(bilstm)

# Save model
torch.save(bilstm.state_dict(), 'bilstm_model.pth')

# Save scaler
import joblib
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved.")
