const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001/api';

export const logUserInteraction = async (userId, action, details = {}) => {
  try {
    const response = await fetch(`${API_BASE_URL}/interactions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userId,
        action,
        details
      })
    });
    
    if (!response.ok) {
      throw new Error('Failed to log interaction');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error logging interaction:', error);
    // Silently fail in production, but log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('Error details:', error);
    }
  }
};

export const getUserInteractions = async (userId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/interactions/user/${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch interactions');
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching interactions:', error);
    return [];
  }
};

// BCI Functions
export const predictAttentionState = async (eegData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/bci/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        eeg_data: eegData
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    
    // Log if using simulated data
    if (result.simulated) {
      console.warn('⚠️ Using simulated BCI data:', result.message);
    }
    
    return result;
  } catch (error) {
    console.error('Error getting BCI prediction:', error);
    
    // Try simulation endpoint as fallback
    try {
      const simulationResponse = await fetch(`${API_BASE_URL}/bci/simulate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (simulationResponse.ok) {
        const simulationResult = await simulationResponse.json();
        console.warn('🔄 Using BCI simulation fallback');
        return {
          ...simulationResult,
          simulated: true,
          message: 'Fallback simulation - BCI service unavailable'
        };
      }
    } catch (simulationError) {
      console.error('Simulation fallback also failed:', simulationError);
    }
    
    // Final fallback to static data
    console.warn('🔴 Using static fallback data');
    return {
      prediction: 'Focused',
      confidence: 0.5,
      probabilities: {
        Focused: 0.5,
        Unfocused: 0.3,
        Drowsy: 0.2
      },
      simulated: true,
      message: 'Static fallback - All BCI services unavailable'
    };
  }
};

// Check BCI service health
export const checkBCIHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/bci/health`);
    return await response.json();
  } catch (error) {
    console.error('BCI health check failed:', error);
    return { status: 'unhealthy', error: error.message };
  }
};

// Get simulated prediction (for testing)
export const getSimulatedPrediction = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/bci/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      throw new Error('Failed to get simulated prediction');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting simulated prediction:', error);
    throw error;
  }
};

// Mock EEG data generator for testing
export const generateMockEEGData = () => {
  // Generate 256 samples of 14-channel EEG data
  const sequenceLength = 256;
  const numChannels = 14;
  const mockData = [];

  for (let i = 0; i < sequenceLength; i++) {
    const sample = [];
    for (let j = 0; j < numChannels; j++) {
      // Generate more realistic EEG-like values with different frequency components
      const baseSignal = Math.sin(2 * Math.PI * 10 * i / sequenceLength); // 10 Hz alpha wave
      const noise = (Math.random() - 0.5) * 20; // Random noise
      const artifact = Math.random() < 0.05 ? (Math.random() - 0.5) * 200 : 0; // Occasional artifacts
      
      sample.push(baseSignal * 50 + noise + artifact);
    }
    mockData.push(sample);
  }

  return mockData;
};

// Generate EEG data with specific characteristics for testing
export const generateTargetedEEGData = (targetState = 'Focused') => {
  const sequenceLength = 256;
  const numChannels = 14;
  const mockData = [];

  for (let i = 0; i < sequenceLength; i++) {
    const sample = [];
    for (let j = 0; j < numChannels; j++) {
      let signal = 0;
      
      switch (targetState) {
        case 'Focused':
          // Higher beta waves (13-30 Hz) for focus
          signal = Math.sin(2 * Math.PI * 20 * i / sequenceLength) * 60;
          break;
        case 'Drowsy':
          // Higher theta waves (4-8 Hz) for drowsiness
          signal = Math.sin(2 * Math.PI * 6 * i / sequenceLength) * 80;
          break;
        case 'Unfocused':
        default:
          // Mixed frequencies for unfocused state
          signal = Math.sin(2 * Math.PI * 12 * i / sequenceLength) * 40 +
                  Math.sin(2 * Math.PI * 8 * i / sequenceLength) * 30;
          break;
      }
      
      const noise = (Math.random() - 0.5) * 15;
      sample.push(signal + noise);
    }
    mockData.push(sample);
  }

  return mockData;
};
