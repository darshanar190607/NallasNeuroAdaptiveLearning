import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useBCI } from '../../contexts/BCIContext';
import { createContentService } from '../../services/contentService';

interface Module {
  type: string;
  title: string;
  duration: string;
  description?: string;
  enhancements?: string[];
}

interface Content {
  title: string;
  description: string;
  modules: Module[];
  studyTips?: string[];
  suggestion?: string;
}

const AdaptiveLearning: React.FC = () => {
  const { user } = useAuth();
  const { bciState, startMonitoring, stopMonitoring, getAdaptiveMessage } = useBCI();
  const [selectedSubject, setSelectedSubject] = useState('mathematics');
  const [content, setContent] = useState<Content | null>(null);
  const [currentModule, setCurrentModule] = useState(0);
  const [isLearning, setIsLearning] = useState(false);

  useEffect(() => {
    if (user?.profile) {
      updateContent();
    }
  }, [user, selectedSubject, bciState]);

  const updateContent = () => {
    if (!user?.profile) return;

    const contentService = createContentService(user.profile, bciState);
    const personalizedContent = contentService.getPersonalizedContent(selectedSubject);
    setContent(personalizedContent);
  };

  const startLearningSession = () => {
    setIsLearning(true);
    setCurrentModule(0);
    if (user?.bciPreferences?.enableBCI) {
      startMonitoring();
    }
  };

  const endLearningSession = () => {
    setIsLearning(false);
    stopMonitoring();
  };

  const nextModule = () => {
    if (content && currentModule < content.modules.length - 1) {
      setCurrentModule(currentModule + 1);
    } else {
      endLearningSession();
    }
  };

  const getModuleIcon = (type: string) => {
    switch (type) {
      case 'video': return '🎥';
      case 'interactive': return '🎮';
      case 'text': return '📖';
      case 'quiz': return '❓';
      case 'audio': return '🎧';
      default: return '📚';
    }
  };

  if (!user) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-400">Please sign in to access personalized learning content.</p>
      </div>
    );
  }

  if (!content) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-500 mx-auto"></div>
        <p className="text-gray-400 mt-4">Loading personalized content...</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* BCI Status */}
      {user.bciPreferences?.enableBCI && (
        <div className="mb-6 p-4 bg-gray-800 rounded-lg border border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${
                bciState.serviceHealth === 'healthy' ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
              <span className="text-sm text-gray-300">
                BCI Status: {bciState.currentState} ({Math.round(bciState.confidence * 100)}%)
              </span>
            </div>
            <div className="text-sm text-orange-400">
              {getAdaptiveMessage()}
            </div>
          </div>
        </div>
      )}

      {/* Subject Selection */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-white mb-4">Choose Your Subject</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {['mathematics', 'science', 'technology'].map(subject => (
            <button
              key={subject}
              onClick={() => setSelectedSubject(subject)}
              className={`p-4 rounded-lg border-2 transition-all ${
                selectedSubject === subject
                  ? 'border-orange-500 bg-orange-500/20 text-white'
                  : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
              }`}
            >
              <div className="text-2xl mb-2">
                {subject === 'mathematics' ? '🔢' : subject === 'science' ? '🔬' : '💻'}
              </div>
              <div className="font-semibold capitalize">{subject}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Content Overview */}
      <div className="mb-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-bold text-white mb-2">{content.title}</h3>
          <p className="text-gray-400 mb-4">{content.description}</p>
          
          {content.suggestion && (
            <div className="mb-4 p-3 bg-blue-500/20 border border-blue-500 rounded-lg">
              <p className="text-blue-300">💡 {content.suggestion}</p>
            </div>
          )}

          {content.studyTips && content.studyTips.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-semibold text-gray-300 mb-2">Study Tips for You:</h4>
              <ul className="text-sm text-gray-400 space-y-1">
                {content.studyTips.map((tip, index) => (
                  <li key={index}>• {tip}</li>
                ))}
              </ul>
            </div>
          )}

          {!isLearning ? (
            <button
              onClick={startLearningSession}
              className="bg-gradient-to-r from-orange-500 to-red-500 text-white px-6 py-3 rounded-lg hover:from-orange-600 hover:to-red-600 transition-all"
            >
              Start Learning Session
            </button>
          ) : (
            <button
              onClick={endLearningSession}
              className="bg-gray-600 text-white px-6 py-3 rounded-lg hover:bg-gray-700 transition-all"
            >
              End Session
            </button>
          )}
        </div>
      </div>

      {/* Learning Modules */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <h4 className="text-lg font-semibold text-white mb-4">Learning Modules</h4>
          <div className="space-y-3">
            {content.modules.map((module, index) => (
              <div
                key={index}
                className={`p-4 rounded-lg border-2 transition-all ${
                  isLearning && index === currentModule
                    ? 'border-orange-500 bg-orange-500/20'
                    : index < currentModule
                    ? 'border-green-500 bg-green-500/20'
                    : 'border-gray-600 bg-gray-700'
                }`}
              >
                <div className="flex items-center space-x-3 mb-2">
                  <span className="text-2xl">{getModuleIcon(module.type)}</span>
                  <div>
                    <h5 className="font-semibold text-white">{module.title}</h5>
                    <p className="text-sm text-gray-400">{module.duration}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="lg:col-span-2">
          {isLearning && content.modules[currentModule] && (
            <div className="bg-gray-800 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-xl font-semibold text-white">
                  {content.modules[currentModule].title}
                </h4>
                <span className="text-sm text-gray-400">
                  Module {currentModule + 1} of {content.modules.length}
                </span>
              </div>

              <div className="mb-6">
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${((currentModule + 1) / content.modules.length) * 100}%` }}
                  ></div>
                </div>
              </div>

              <div className="bg-gray-700 rounded-lg p-8 mb-6 text-center">
                <div className="text-6xl mb-4">
                  {getModuleIcon(content.modules[currentModule].type)}
                </div>
                <h5 className="text-lg font-semibold text-white mb-2">
                  {content.modules[currentModule].title}
                </h5>
                <p className="text-gray-400 mb-4">
                  Interactive learning content would be displayed here.
                </p>
                <p className="text-sm text-gray-500">
                  Duration: {content.modules[currentModule].duration}
                </p>
              </div>

              <div className="flex justify-between">
                <button
                  onClick={() => setCurrentModule(Math.max(0, currentModule - 1))}
                  disabled={currentModule === 0}
                  className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  Previous
                </button>
                
                <button
                  onClick={nextModule}
                  className="px-6 py-2 bg-gradient-to-r from-orange-500 to-red-500 text-white rounded-lg hover:from-orange-600 hover:to-red-600 transition-all"
                >
                  {currentModule === content.modules.length - 1 ? 'Complete' : 'Next'}
                </button>
              </div>
            </div>
          )}

          {!isLearning && (
            <div className="bg-gray-800 rounded-lg p-8 text-center">
              <div className="text-4xl mb-4">🎯</div>
              <h4 className="text-xl font-semibold text-white mb-2">Ready to Learn?</h4>
              <p className="text-gray-400">
                Start your personalized learning session to begin with content adapted to your learning style.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdaptiveLearning;