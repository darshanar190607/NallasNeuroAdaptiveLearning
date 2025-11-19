# 🧠 NeuroAdaptive EdTech - BCI Integration Guide

## Overview

This project integrates a Brain-Computer Interface (BCI) system with a React-based educational platform. The system uses EEG data to detect attention states (Focused, Unfocused, Drowsy) and adapts the learning experience in real-time.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React App     │    │   Node.js API   │    │  FastAPI BCI    │
│   (Port 3000)   │◄──►│   (Port 5000)   │◄──►│   (Port 8000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BCI Context   │    │   MongoDB       │    │  PyTorch Model  │
│   & Components  │    │   Database      │    │   + Scaler      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Components

### 1. FastAPI BCI Server (`fastapi_server.py`)
- **Purpose**: Handles BCI model predictions
- **Port**: 8000
- **Features**:
  - CNN-BiLSTM model for EEG classification
  - Real-time attention state prediction
  - Health monitoring endpoints
  - Simulation endpoints for testing
  - CORS enabled for frontend integration

### 2. Node.js API Server (`server/server.js`)
- **Purpose**: Main API gateway and database operations
- **Port**: 5000
- **Features**:
  - Routes BCI requests to FastAPI server
  - Handles user interactions and logging
  - MongoDB integration
  - Fallback mechanisms for BCI service

### 3. React Frontend
- **Purpose**: User interface with adaptive learning features
- **Port**: 3000
- **Key Components**:
  - `BCIContext`: Global state management for BCI data
  - `BCIStatus`: Real-time monitoring component
  - Adaptive UI based on attention states

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
# Run the batch script
start_servers.bat
```

**macOS/Linux:**
```bash
# Make script executable and run
chmod +x start_servers.sh
./start_servers.sh
```

### Option 2: Manual Setup

1. **Install Python Dependencies:**
```bash
pip install -r requirements_fastapi.txt
```

2. **Extract BCI Model:**
```bash
python extract_model.py
```

3. **Start FastAPI Server:**
```bash
python fastapi_server.py
```

4. **Start Node.js Server:**
```bash
cd server
npm install
npm start
```

5. **Start React App:**
```bash
npm install
npm start
```

## API Endpoints

### FastAPI BCI Server (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service status |
| `/health` | GET | Health check |
| `/predict` | POST | EEG prediction |
| `/simulate` | POST | Simulated prediction |

### Node.js API Server (Port 5000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/bci/predict` | POST | BCI prediction (proxy) |
| `/api/bci/health` | GET | BCI health check |
| `/api/bci/simulate` | POST | BCI simulation |
| `/api/interactions` | POST | Log user interaction |

## BCI Data Format

### Input (EEG Data)
```json
{
  "eeg_data": [
    [ch1, ch2, ch3, ..., ch14],  // Sample 1
    [ch1, ch2, ch3, ..., ch14],  // Sample 2
    ...                          // 256 samples total
  ]
}
```

### Output (Prediction)
```json
{
  "prediction": "Focused",
  "confidence": 0.85,
  "probabilities": {
    "Focused": 0.85,
    "Unfocused": 0.10,
    "Drowsy": 0.05
  },
  "simulated": false,
  "message": "Real-time prediction"
}
```

## Features

### 🧠 Real-time BCI Monitoring
- Continuous EEG data processing
- Attention state classification
- Confidence scoring
- Visual feedback in UI

### 🎮 Adaptive Learning Interface
- Dynamic content adaptation based on attention state
- Visual effects that respond to cognitive load
- Personalized learning recommendations

### 🔧 Testing & Simulation
- Mock EEG data generation
- Targeted state simulation
- Health monitoring
- Fallback mechanisms

### 📊 Data Visualization
- Real-time probability bars
- State history tracking
- Performance metrics

## Usage in React Components

### Using BCI Context
```tsx
import { useBCI } from '../contexts/BCIContext';

function MyComponent() {
  const { bciState, startMonitoring, getAdaptiveStyles } = useBCI();
  
  return (
    <div style={getAdaptiveStyles()}>
      <p>Current State: {bciState.currentState}</p>
      <p>Confidence: {(bciState.confidence * 100).toFixed(0)}%</p>
      <button onClick={startMonitoring}>Start Monitoring</button>
    </div>
  );
}
```

### Adaptive Styling
The BCI context provides adaptive styles based on attention state:
- **Focused**: Enhanced brightness and slight scaling
- **Unfocused**: Reduced opacity and slight blur
- **Drowsy**: Grayscale filter and reduced opacity

## Troubleshooting

### Common Issues

1. **FastAPI Server Won't Start**
   - Check Python dependencies: `pip install -r requirements_fastapi.txt`
   - Verify port 8000 is available
   - Run model extraction: `python extract_model.py`

2. **BCI Predictions Failing**
   - Check FastAPI server health: `http://localhost:8000/health`
   - Verify EEG data format (256 samples × 14 channels)
   - Use simulation endpoint for testing

3. **React App Not Connecting**
   - Ensure all servers are running
   - Check CORS configuration
   - Verify API URLs in environment variables

### Debug Mode

Enable debug logging by setting environment variables:
```bash
# For FastAPI
export DEBUG=1

# For Node.js
export NODE_ENV=development
```

## Model Information

### CNN-BiLSTM Architecture
- **Input**: 256 samples × 14 EEG channels
- **CNN Layers**: Feature extraction from EEG signals
- **BiLSTM Layers**: Temporal pattern recognition
- **Output**: 3-class classification (Focused/Unfocused/Drowsy)

### Training Data
- Based on EEG attention state detection dataset
- 14-channel EEG recordings
- Preprocessed with StandardScaler
- Trained for cognitive state classification

## Development

### Adding New Features

1. **New BCI Endpoints**: Add to `fastapi_server.py`
2. **UI Components**: Use `BCIContext` for state management
3. **Adaptive Behaviors**: Extend `getAdaptiveStyles()` and `getAdaptiveMessage()`

### Testing

```bash
# Test BCI service
curl -X POST http://localhost:8000/simulate

# Test Node.js proxy
curl -X POST http://localhost:5000/api/bci/simulate

# Health checks
curl http://localhost:8000/health
curl http://localhost:5000/api/bci/health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is part of the NeuroAdaptive EdTech platform and follows the same licensing terms.

---

For more information, see the main project README or contact the development team.