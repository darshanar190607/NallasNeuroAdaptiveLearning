import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { predictAttentionState, generateMockEEGData, generateTargetedEEGData, checkBCIHealth } from '../utils/api';

export type AttentionState = 'Focused' | 'Unfocused' | 'Drowsy';

interface BCIState {
  currentState: AttentionState;
  confidence: number;
  probabilities: {
    Focused: number;
    Unfocused: number;
    Drowsy: number;
  };
  isMonitoring: boolean;
  lastUpdate: Date | null;
  isSimulated: boolean;
  serviceHealth: 'healthy' | 'unhealthy' | 'unknown';
  message?: string;
}

interface BCIContextType {
  bciState: BCIState;
  startMonitoring: () => void;
  stopMonitoring: () => void;
  getAdaptiveStyles: () => React.CSSProperties;
  getAdaptiveMessage: () => string;
  checkHealth: () => Promise<void>;
  simulateState: (state: AttentionState) => void;
}

const BCIContext = createContext<BCIContextType | undefined>(undefined);

export const useBCI = () => {
  const context = useContext(BCIContext);
  if (!context) {
    throw new Error('useBCI must be used within a BCIProvider');
  }
  return context;
};

interface BCIProviderProps {
  children: ReactNode;
}

export const BCIProvider: React.FC<BCIProviderProps> = ({ children }) => {
  const [bciState, setBciState] = useState<BCIState>({
    currentState: 'Focused',
    confidence: 0.5,
    probabilities: {
      Focused: 0.5,
      Unfocused: 0.3,
      Drowsy: 0.2
    },
    isMonitoring: false,
    lastUpdate: null,
    isSimulated: false,
    serviceHealth: 'unknown'
  });

  const updateAttentionState = async () => {
    if (!bciState.isMonitoring) return;

    try {
      // Generate mock EEG data for testing
      const mockEEGData = generateMockEEGData();

      // Get prediction from BCI service
      const prediction = await predictAttentionState(mockEEGData);

      setBciState(prev => ({
        ...prev,
        currentState: prediction.prediction as AttentionState,
        confidence: prediction.confidence,
        probabilities: prediction.probabilities,
        lastUpdate: new Date(),
        isSimulated: prediction.simulated || false,
        message: prediction.message,
        serviceHealth: 'healthy'
      }));
    } catch (error) {
      console.error('Failed to update attention state:', error);
      setBciState(prev => ({
        ...prev,
        serviceHealth: 'unhealthy',
        message: 'BCI service connection failed'
      }));
    }
  };

  const checkHealth = async () => {
    try {
      const health = await checkBCIHealth();
      setBciState(prev => ({
        ...prev,
        serviceHealth: health.status === 'healthy' ? 'healthy' : 'unhealthy'
      }));
    } catch (error) {
      setBciState(prev => ({
        ...prev,
        serviceHealth: 'unhealthy'
      }));
    }
  };

  const simulateState = (targetState: AttentionState) => {
    const mockEEGData = generateTargetedEEGData(targetState);
    
    // Create a simulated prediction
    const baseProb = 0.7;
    const otherProb = (1 - baseProb) / 2;
    
    const probabilities = {
      Focused: targetState === 'Focused' ? baseProb : otherProb,
      Unfocused: targetState === 'Unfocused' ? baseProb : otherProb,
      Drowsy: targetState === 'Drowsy' ? baseProb : otherProb
    };
    
    setBciState(prev => ({
      ...prev,
      currentState: targetState,
      confidence: baseProb,
      probabilities,
      lastUpdate: new Date(),
      isSimulated: true,
      message: `Simulated ${targetState} state for testing`
    }));
  };

  const startMonitoring = () => {
    setBciState(prev => ({ ...prev, isMonitoring: true }));
  };

  const stopMonitoring = () => {
    setBciState(prev => ({ ...prev, isMonitoring: false }));
  };

  // Monitor attention state every 5 seconds when active
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (bciState.isMonitoring) {
      updateAttentionState(); // Initial update
      interval = setInterval(updateAttentionState, 5000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [bciState.isMonitoring]);

  // Check health on component mount
  useEffect(() => {
    checkHealth();
  }, []);

  const getAdaptiveStyles = (): React.CSSProperties => {
    const { currentState, confidence } = bciState;

    // Base styles
    const baseStyles: React.CSSProperties = {
      transition: 'all 0.5s ease-in-out'
    };

    // Adaptive styles based on attention state
    switch (currentState) {
      case 'Focused':
        return {
          ...baseStyles,
          filter: `brightness(${1 + confidence * 0.2})`,
          transform: `scale(${1 + confidence * 0.05})`
        };
      case 'Unfocused':
        return {
          ...baseStyles,
          filter: `blur(${1 - confidence * 0.5}px)`,
          opacity: 0.8 + confidence * 0.2
        };
      case 'Drowsy':
        return {
          ...baseStyles,
          filter: `grayscale(${0.3 - confidence * 0.3})`,
          opacity: 0.6 + confidence * 0.4
        };
      default:
        return baseStyles;
    }
  };

  const getAdaptiveMessage = (): string => {
    const { currentState, confidence } = bciState;

    if (confidence < 0.6) {
      return "Calibrating attention monitoring...";
    }

    switch (currentState) {
      case 'Focused':
        return confidence > 0.8
          ? "Excellent focus! You're in the zone."
          : "Good focus. Keep it up!";
      case 'Unfocused':
        return confidence > 0.8
          ? "Attention wandering. Try taking a short break."
          : "Slight distraction detected.";
      case 'Drowsy':
        return confidence > 0.8
          ? "Drowsiness detected. Consider a quick walk or coffee."
          : "Feeling tired? Take a moment to refresh.";
      default:
        return "Monitoring your attention state...";
    }
  };

  const value: BCIContextType = {
    bciState,
    startMonitoring,
    stopMonitoring,
    getAdaptiveStyles,
    getAdaptiveMessage,
    checkHealth,
    simulateState
  };

  return (
    <BCIContext.Provider value={value}>
      {children}
    </BCIContext.Provider>
  );
};