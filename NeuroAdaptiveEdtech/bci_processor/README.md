# BCI Processor

A Python package for processing BCI data from both simulated and real devices.

## Project Structure

```
bci_processor/
├── bci_processor/             # Main package
│   ├── __init__.py
│   ├── core/                  # Core processing modules
│   │   ├── __init__.py
│   │   ├── preprocessor.py    # Data preprocessing
│   │   ├── feature_extractor.py # Feature extraction
│   │   └── classifier.py      # ML model for state classification
│   │
│   ├── devices/               # Device interfaces
│   │   ├── __init__.py
│   │   ├── base_device.py     # Abstract base class for all devices
│   │   ├── matlab_device.py   # MATLAB file interface
│   │   └── muse_device.py     # Example for Muse headset
│   │
│   ├── models/                # Trained models
│   │   └── __init__.py
│   │
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── config.py          # Configuration settings
│       └── visualization.py   # Data visualization
│
├── data/                      # Data directory
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed data files
│
├── notebooks/                 # Jupyter notebooks
│   └── BCI.ipynb             # Your existing notebook
│
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
└── setup.py                  # Package installation
```

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Using with MATLAB files
```python
from bci_processor.devices import MatlabDevice
from bci_processor.core import BCIProcessor

# Initialize with MATLAB file
device = MatlabDevice('path/to/your/data.mat')

# Create processor
processor = BCIProcessor(device)

# Get current state
state = processor.get_current_state()
print(f"Current cognitive state: {state}")
```

### Using with Real BCI Device (Example: Muse)
```python
from bci_processor.devices import MuseDevice
from bci_processor.core import BCIProcessor

# Initialize with real device
device = MuseDevice()  # Connect to Muse headset

# Create processor
processor = BCIProcessor(device)

# Start real-time processing
processor.start_stream()

# Get current state (non-blocking)
state = processor.get_current_state()
print(f"Current cognitive state: {state}")

# Remember to stop the stream when done
processor.stop_stream()
```

## Adding New Devices

1. Create a new file in `bci_processor/devices/`
2. Extend the `BaseDevice` class
3. Implement the required methods for your device
4. Update `__init__.py` to expose your new device

## Testing

Run tests with:
```bash
pytest tests/
```
