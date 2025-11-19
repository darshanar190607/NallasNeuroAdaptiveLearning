from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

class BaseDevice(ABC):
    """
    Abstract base class for all BCI devices.
    
    All device implementations should inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the BCI device.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the BCI device."""
        pass
    
    @abstractmethod
    def start_stream(self) -> bool:
        """
        Start streaming data from the device.
        
        Returns:
            bool: True if streaming started successfully
        """
        pass
    
    @abstractmethod
    def stop_stream(self) -> None:
        """Stop streaming data from the device."""
        pass
    
    @abstractmethod
    def get_data(self, duration: float = 1.0) -> Dict[str, Any]:
        """
        Get the most recent data from the device.
        
        Args:
            duration: Duration of data to retrieve in seconds
            
        Returns:
            Dictionary containing the data in the format:
            {
                'eeg': np.ndarray,  # Shape: (n_channels, n_samples)
                'sample_rate': float,
                'channels': List[str],  # Channel names
                'timestamps': np.ndarray  # Timestamps for each sample
            }
        """
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the connected device.
        
        Returns:
            Dictionary containing device information
        """
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the device is connected."""
        pass
    
    @property
    @abstractmethod
    def is_streaming(self) -> bool:
        """Check if the device is currently streaming data."""
        pass
