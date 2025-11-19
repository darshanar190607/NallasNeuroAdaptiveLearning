from typing import Dict, Any, Optional
import numpy as np
from ..devices.base_device import BaseDevice
from .preprocessor import Preprocessor
from .feature_extractor import FeatureExtractor
from .classifier import Classifier

class BCIProcessor:
    """
    Main BCI processing pipeline that coordinates data flow from devices through
    processing to classification.
    """
    
    def __init__(self, device: BaseDevice, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BCI processor with a device and optional configuration.
        
        Args:
            device: An instance of a device that implements the BaseDevice interface
            config: Optional configuration dictionary
        """
        self.device = device
        self.config = config or {}
        
        # Initialize processing components
        self.preprocessor = Preprocessor(
            sample_rate=self.device.sample_rate if hasattr(device, 'sample_rate') else 250.0,
            **self.config.get('preprocessing', {})
        )
        
        self.feature_extractor = FeatureExtractor(
            sample_rate=self.preprocessor.sample_rate,
            **self.config.get('feature_extraction', {})
        )
        
        self.classifier = Classifier(**self.config.get('classification', {}))
        
        # State tracking
        self._is_processing = False
        self._current_state = None
        self._state_history = []
    
    def start(self) -> bool:
        """Start the BCI processing pipeline."""
        if not self.device.is_connected:
            if not self.device.connect():
                return False
        
        if not self.device.is_streaming:
            if not self.device.start_stream():
                return False
        
        self._is_processing = True
        return True
    
    def stop(self) -> None:
        """Stop the BCI processing pipeline."""
        self._is_processing = False
        if self.device.is_streaming:
            self.device.stop_stream()
    
    def process_next(self) -> Dict[str, Any]:
        """
        Process the next chunk of data from the device.
        
        Returns:
            Dictionary containing processing results
        """
        if not self._is_processing:
            raise RuntimeError("Processor is not running. Call start() first.")
        
        # Get raw data from device
        raw_data = self.device.get_data()
        
        # Preprocess the data
        processed_data = self.preprocessor.process(
            raw_data['eeg'],
            sample_rate=raw_data['sample_rate']
        )
        
        # Extract features
        features = self.feature_extractor.extract(processed_data)
        
        # Classify the current state
        state = self.classifier.predict(features)
        
        # Update state
        self._current_state = {
            'state': state,
            'timestamp': raw_data.get('timestamps', [0])[-1],
            'features': features,
            'raw_data': raw_data
        }
        self._state_history.append(self._current_state)
        
        return self._current_state
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the most recent state classification."""
        return self._current_state
    
    def get_state_history(self) -> list:
        """Get the history of state classifications."""
        return self._state_history
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the connected device."""
        return self.device.get_device_info()
    
    def is_processing(self) -> bool:
        """Check if the processor is currently running."""
        return self._is_processing
    
    def reset(self) -> None:
        """Reset the processor state."""
        self._current_state = None
        self._state_history = []
        self.preprocessor.reset()
        self.feature_extractor.reset()
        self.classifier.reset()
