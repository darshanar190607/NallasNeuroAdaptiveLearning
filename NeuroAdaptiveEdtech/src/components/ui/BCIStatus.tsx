import React from 'react';
import { useBCI } from '../../contexts/BCIContext';

const BCIStatus: React.FC = () => {
  const { bciState, startMonitoring, stopMonitoring, getAdaptiveMessage, checkHealth, simulateState } = useBCI();

  const getStateColor = (state: string) => {
    switch (state) {
      case 'Focused':
        return 'text-green-400 bg-green-400/20';
      case 'Unfocused':
        return 'text-yellow-400 bg-yellow-400/20';
      case 'Drowsy':
        return 'text-red-400 bg-red-400/20';
      default:
        return 'text-gray-400 bg-gray-400/20';
    }
  };

  const getStateIcon = (state: string) => {
    switch (state) {
      case 'Focused':
        return '🧠';
      case 'Unfocused':
        return '🤔';
      case 'Drowsy':
        return '😴';
      default:
        return '📊';
    }
  };

  return (
    <div className="fixed top-4 right-4 z-50">
      <div className="bg-gray-800/90 backdrop-blur-sm border border-gray-700 rounded-lg p-4 shadow-lg min-w-[280px]">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <h3 className="text-sm font-semibold text-gray-200">BCI Monitor</h3>
            <div className={`w-2 h-2 rounded-full ${
              bciState.serviceHealth === 'healthy' ? 'bg-green-400' :
              bciState.serviceHealth === 'unhealthy' ? 'bg-red-400' : 'bg-yellow-400'
            }`} title={`Service status: ${bciState.serviceHealth}`}></div>
          </div>
          <div className="flex space-x-1">
            <button
              onClick={checkHealth}
              className="px-2 py-1 text-xs rounded bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 transition-colors"
              title="Check service health"
            >
              🔄
            </button>
            <button
              onClick={bciState.isMonitoring ? stopMonitoring : startMonitoring}
              className={`px-3 py-1 text-xs rounded-full transition-colors ${
                bciState.isMonitoring
                  ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                  : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
              }`}
            >
              {bciState.isMonitoring ? 'Stop' : 'Start'}
            </button>
          </div>
        </div>

        {bciState.isMonitoring && (
          <>
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-lg">{getStateIcon(bciState.currentState)}</span>
              <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStateColor(bciState.currentState)}`}>
                {bciState.currentState}
                {bciState.isSimulated && ' 🧪'}
              </span>
              <span className="text-xs text-gray-400">
                {(bciState.confidence * 100).toFixed(0)}%
              </span>
            </div>

            <div className="space-y-1 mb-3">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Focused</span>
                <span className="text-gray-300">{(bciState.probabilities.Focused * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1">
                <div
                  className="bg-green-400 h-1 rounded-full transition-all duration-500"
                  style={{ width: `${bciState.probabilities.Focused * 100}%` }}
                ></div>
              </div>

              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Unfocused</span>
                <span className="text-gray-300">{(bciState.probabilities.Unfocused * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1">
                <div
                  className="bg-yellow-400 h-1 rounded-full transition-all duration-500"
                  style={{ width: `${bciState.probabilities.Unfocused * 100}%` }}
                ></div>
              </div>

              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Drowsy</span>
                <span className="text-gray-300">{(bciState.probabilities.Drowsy * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1">
                <div
                  className="bg-red-400 h-1 rounded-full transition-all duration-500"
                  style={{ width: `${bciState.probabilities.Drowsy * 100}%` }}
                ></div>
              </div>
            </div>

            <p className="text-xs text-gray-300 leading-relaxed">
              {getAdaptiveMessage()}
            </p>

            {bciState.message && (
              <p className="text-xs text-blue-300 mt-1 italic">
                {bciState.message}
              </p>
            )}

            {/* Simulation Controls */}
            <div className="mt-3 pt-2 border-t border-gray-600">
              <p className="text-xs text-gray-400 mb-2">Test States:</p>
              <div className="flex space-x-1">
                <button
                  onClick={() => simulateState('Focused')}
                  className="px-2 py-1 text-xs rounded bg-green-500/20 text-green-400 hover:bg-green-500/30 transition-colors"
                >
                  Focus
                </button>
                <button
                  onClick={() => simulateState('Unfocused')}
                  className="px-2 py-1 text-xs rounded bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 transition-colors"
                >
                  Unfocus
                </button>
                <button
                  onClick={() => simulateState('Drowsy')}
                  className="px-2 py-1 text-xs rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                >
                  Drowsy
                </button>
              </div>
            </div>

            {bciState.lastUpdate && (
              <p className="text-xs text-gray-500 mt-2">
                Last update: {bciState.lastUpdate.toLocaleTimeString()}
              </p>
            )}
          </>
        )}

        {!bciState.isMonitoring && (
          <p className="text-xs text-gray-400">
            Click "Start" to begin monitoring your attention state
          </p>
        )}
      </div>
    </div>
  );
};

export default BCIStatus;