import os
import numpy as np
from typing import Dict, Any, Optional
import scipy.io as sio
from pathlib import Path
from datetime import datetime

from .base_device import BaseDevice

class MatlabDevice(BaseDevice):
    """
    A device interface for loading and simulating BCI data from MATLAB .mat files.
    
    This class simulates a real-time BCI device by reading pre-recorded data
    from a MATLAB .mat file and streaming it in real-time.
    """
    
    def __init__(self, file_path: str, sample_rate: float = 250.0, loop: bool = True):
        """
        Initialize the MATLAB device interface.
        
        Args:
            file_path: Path to the .mat file containing BCI data
            sample_rate: Sample rate in Hz (used if not specified in the data)
            loop: If True, loop the data when the end is reached
        """
        self.file_path = Path(file_path)
        self.sample_rate = sample_rate
        self.loop = loop
        
        # Device state
        self._connected = False
        self._streaming = False
        self._current_sample = 0
        
        # Data storage
        self.data: Optional[Dict[str, Any]] = None
        self.channel_names: Optional[list] = None
        self.timestamps: Optional[np.ndarray] = None
        self.n_samples: int = 0
        
    def connect(self) -> bool:
        """Load the MATLAB file and prepare for streaming."""
        try:
            # Load the MATLAB file
            mat_data = sio.loadmat(self.file_path)
            
            # Extract data - adjust these keys based on your .mat file structure
            if 'eeg_data' in mat_data:
                self.data = mat_data['eeg_data']  # Expected shape: (n_channels, n_samples)
            elif 'data' in mat_data:
                self.data = mat_data['data']
            else:
                # Try to find the first 2D array in the .mat file
                for key, value in mat_data.items():
                    if isinstance(value, np.ndarray) and value.ndim == 2:
                        self.data = value
                        break
            
            if self.data is None:
                raise ValueError("No suitable data array found in the .mat file")
            
            # Transpose if needed (channels should be first dimension)
            if self.data.shape[0] > self.data.shape[1]:
                self.data = self.data.T
            
            # Get channel names if available
            if 'channels' in mat_data:
                self.channel_names = [str(c[0][0]) for c in mat_data['channels']]
            else:
                self.channel_names = [f'Channel_{i+1}' for i in range(self.data.shape[0])]
            
            # Get sample rate if available
            if 'srate' in mat_data:
                self.sample_rate = float(mat_data['srate'].flatten()[0])
            
            self.n_samples = self.data.shape[1]
            self._current_sample = 0
            self._connected = True
            
            return True
            
        except Exception as e:
            print(f"Error loading MATLAB file: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect the device and clean up resources."""
        self._connected = False
        self._streaming = False
        self.data = None
        self.channel_names = None
    
    def start_stream(self) -> bool:
        """Start streaming data from the MATLAB file."""
        if not self._connected:
            if not self.connect():
                return False
        
        self._streaming = True
        self._start_time = datetime.now()
        return True
    
    def stop_stream(self) -> None:
        """Stop streaming data."""
        self._streaming = False
    
    def get_data(self, duration: float = 1.0) -> Dict[str, Any]:
        """
        Get a chunk of data from the MATLAB file.
        
        Args:
            duration: Duration of data to retrieve in seconds
            
        Returns:
            Dictionary containing the data
        """
        if not self._streaming or self.data is None:
            raise RuntimeError("Device is not streaming")
        
        n_samples = int(duration * self.sample_rate)
        start_sample = self._current_sample
        end_sample = start_sample + n_samples
        
        # Handle end of data
        if end_sample >= self.n_samples:
            if self.loop:
                end_sample = self.n_samples - 1
                self._current_sample = 0
            else:
                end_sample = self.n_samples - 1
                self._streaming = False
        else:
            self._current_sample = end_sample
        
        # Get the data slice
        data_slice = self.data[:, start_sample:end_sample]
        
        # Generate timestamps
        timestamps = np.linspace(
            start_sample / self.sample_rate,
            end_sample / self.sample_rate,
            num=end_sample - start_sample,
            endpoint=False
        )
        
        return {
            'eeg': data_slice,
            'sample_rate': self.sample_rate,
            'channels': self.channel_names,
            'timestamps': timestamps
        }
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the MATLAB device."""
        return {
            'device_type': 'MATLAB File',
            'file_path': str(self.file_path),
            'sample_rate': self.sample_rate,
            'n_channels': len(self.channel_names) if self.channel_names else 0,
            'n_samples': self.n_samples,
            'channels': self.channel_names,
            'connected': self._connected,
            'streaming': self._streaming
        }
    
    @property
    def is_connected(self) -> bool:
        """Check if the device is connected."""
        return self._connected
    
    @property
    def is_streaming(self) -> bool:
        """Check if the device is currently streaming data."""
        return self._streaming
