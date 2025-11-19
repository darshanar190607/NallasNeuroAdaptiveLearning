import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';

interface SurveyData {
  ageGroup: string;
  educationLevel: string;
  learningStyle: string;
  preferredPace: string;
  primaryInterests: string[];
  careerGoals: string[];
  learningChallenges: string[];
  accommodationsNeeded: string[];
  deviceComfort: string;
  preferredContentTypes: string[];
  studyEnvironment: string;
  preferredStudyTime: string;
  sessionDuration: string;
  breakFrequency: string;
  bciInterest: string;
  adaptationComfort: string;
  privacyConcerns: string;
}

const PreSurvey: React.FC<{ onComplete: () => void }> = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [surveyData, setSurveyData] = useState<SurveyData>({
    ageGroup: '',
    educationLevel: '',
    learningStyle: '',
    preferredPace: '',
    primaryInterests: [],
    careerGoals: [],
    learningChallenges: [],
    accommodationsNeeded: [],
    deviceComfort: '',
    preferredContentTypes: [],
    studyEnvironment: '',
    preferredStudyTime: '',
    sessionDuration: '',
    breakFrequency: '',
    bciInterest: '',
    adaptationComfort: '',
    privacyConcerns: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { token, updateUser } = useAuth();

  const handleMultiSelect = (field: keyof SurveyData, value: string) => {
    const currentValues = surveyData[field] as string[];
    const newValues = currentValues.includes(value)
      ? currentValues.filter(v => v !== value)
      : [...currentValues, value];
    
    setSurveyData({ ...surveyData, [field]: newValues });
  };

  const handleSingleSelect = (field: keyof SurveyData, value: string) => {
    setSurveyData({ ...surveyData, [field]: value });
  };

  const submitSurvey = async () => {
    setIsSubmitting(true);
    try {
      const response = await fetch('http://localhost:5000/api/survey/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ responses: surveyData })
      });

      if (response.ok) {
        const data = await response.json();
        updateUser({ needsSurvey: false, profile: data.profile });
        onComplete();
      } else {
        throw new Error('Failed to submit survey');
      }
    } catch (error) {
      console.error('Survey submission error:', error);
      alert('Failed to submit survey. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const steps = [
    {
      title: "Basic Information",
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">What's your age group?</h3>
            <div className="grid grid-cols-1 gap-3">
              {['12-18', '18-22', '22+'].map(option => (
                <button
                  key={option}
                  onClick={() => handleSingleSelect('ageGroup', option)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    surveyData.ageGroup === option
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {option} years old
                </button>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Education Level</h3>
            <div className="grid grid-cols-1 gap-3">
              {[
                { value: 'middle-school', label: 'Middle School' },
                { value: 'high-school', label: 'High School' },
                { value: 'undergraduate', label: 'Undergraduate' },
                { value: 'graduate', label: 'Graduate' },
                { value: 'professional', label: 'Professional' }
              ].map(option => (
                <button
                  key={option.value}
                  onClick={() => handleSingleSelect('educationLevel', option.value)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    surveyData.educationLevel === option.value
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Learning Preferences",
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">How do you learn best?</h3>
            <div className="grid grid-cols-1 gap-3">
              {[
                { value: 'visual', label: 'Visual (images, diagrams, videos)' },
                { value: 'auditory', label: 'Auditory (listening, discussions)' },
                { value: 'kinesthetic', label: 'Kinesthetic (hands-on, movement)' },
                { value: 'reading-writing', label: 'Reading/Writing (text-based)' },
                { value: 'mixed', label: 'Mixed (combination of styles)' }
              ].map(option => (
                <button
                  key={option.value}
                  onClick={() => handleSingleSelect('learningStyle', option.value)}
                  className={`p-3 rounded-lg border-2 transition-all text-left ${
                    surveyData.learningStyle === option.value
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Preferred Learning Pace</h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { value: 'slow', label: 'Slow & Steady' },
                { value: 'moderate', label: 'Moderate' },
                { value: 'fast', label: 'Fast-paced' },
                { value: 'adaptive', label: 'Adaptive' }
              ].map(option => (
                <button
                  key={option.value}
                  onClick={() => handleSingleSelect('preferredPace', option.value)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    surveyData.preferredPace === option.value
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Interests & Goals",
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">What are your primary interests? (Select all that apply)</h3>
            <div className="grid grid-cols-2 gap-3">
              {['STEM', 'Arts', 'Business', 'Languages', 'Sports', 'Technology', 'Science', 'Mathematics', 'History', 'Literature'].map(interest => (
                <button
                  key={interest}
                  onClick={() => handleMultiSelect('primaryInterests', interest)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    surveyData.primaryInterests.includes(interest)
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {interest}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Career Goals (Optional)</h3>
            <textarea
              value={surveyData.careerGoals.join('\n')}
              onChange={(e) => setSurveyData({ ...surveyData, careerGoals: e.target.value.split('\n').filter(g => g.trim()) })}
              placeholder="Enter your career goals, one per line..."
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-orange-500"
              rows={4}
            />
          </div>
        </div>
      )
    },
    {
      title: "Learning Challenges",
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Do you have any learning challenges? (Select all that apply)</h3>
            <div className="grid grid-cols-2 gap-3">
              {['ADHD', 'Dyslexia', 'Anxiety', 'Focus Issues', 'Memory Issues', 'Processing Speed', 'None'].map(challenge => (
                <button
                  key={challenge}
                  onClick={() => handleMultiSelect('learningChallenges', challenge)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    surveyData.learningChallenges.includes(challenge)
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {challenge}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">What accommodations help you learn better?</h3>
            <div className="grid grid-cols-1 gap-3">
              {['Extended Time', 'Frequent Breaks', 'Visual Aids', 'Audio Support', 'Simplified Instructions', 'None'].map(accommodation => (
                <button
                  key={accommodation}
                  onClick={() => handleMultiSelect('accommodationsNeeded', accommodation)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    surveyData.accommodationsNeeded.includes(accommodation)
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {accommodation}
                </button>
              ))}
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Study Habits",
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Preferred Study Environment</h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { value: 'quiet-room', label: 'Quiet Room' },
                { value: 'background-music', label: 'Background Music' },
                { value: 'nature-sounds', label: 'Nature Sounds' },
                { value: 'collaborative-space', label: 'Collaborative Space' }
              ].map(option => (
                <button
                  key={option.value}
                  onClick={() => handleSingleSelect('studyEnvironment', option.value)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    surveyData.studyEnvironment === option.value
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Best Study Time</h3>
            <div className="grid grid-cols-3 gap-3">
              {[
                { value: 'early-morning', label: 'Early Morning' },
                { value: 'morning', label: 'Morning' },
                { value: 'afternoon', label: 'Afternoon' },
                { value: 'evening', label: 'Evening' },
                { value: 'night', label: 'Night' }
              ].map(option => (
                <button
                  key={option.value}
                  onClick={() => handleSingleSelect('preferredStudyTime', option.value)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    surveyData.preferredStudyTime === option.value
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Preferred Session Duration</h3>
            <div className="grid grid-cols-3 gap-3">
              {['15-30min', '30-45min', '45-60min', '60-90min', '90min+'].map(duration => (
                <button
                  key={duration}
                  onClick={() => handleSingleSelect('sessionDuration', duration)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    surveyData.sessionDuration === duration
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {duration}
                </button>
              ))}
            </div>
          </div>
        </div>
      )
    },
    {
      title: "BCI & Adaptive Features",
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Interest in Brain-Computer Interface (BCI) Features</h3>
            <p className="text-gray-400 mb-4">BCI technology can monitor your attention and adapt content in real-time</p>
            <div className="grid grid-cols-2 gap-3">
              {[
                { value: 'very-interested', label: 'Very Interested' },
                { value: 'somewhat-interested', label: 'Somewhat Interested' },
                { value: 'neutral', label: 'Neutral' },
                { value: 'not-interested', label: 'Not Interested' }
              ].map(option => (
                <button
                  key={option.value}
                  onClick={() => handleSingleSelect('bciInterest', option.value)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    surveyData.bciInterest === option.value
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Comfort with Adaptive Content</h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { value: 'minimal', label: 'Minimal Changes' },
                { value: 'moderate', label: 'Moderate Adaptation' },
                { value: 'aggressive', label: 'Aggressive Adaptation' },
                { value: 'full-automation', label: 'Full Automation' }
              ].map(option => (
                <button
                  key={option.value}
                  onClick={() => handleSingleSelect('adaptationComfort', option.value)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    surveyData.adaptationComfort === option.value
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Privacy Concerns</h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { value: 'none', label: 'No Concerns' },
                { value: 'minimal', label: 'Minimal Concerns' },
                { value: 'moderate', label: 'Moderate Concerns' },
                { value: 'high', label: 'High Concerns' }
              ].map(option => (
                <button
                  key={option.value}
                  onClick={() => handleSingleSelect('privacyConcerns', option.value)}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    surveyData.privacyConcerns === option.value
                      ? 'border-orange-500 bg-orange-500/20 text-white'
                      : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      )
    }
  ];

  const isStepComplete = (stepIndex: number) => {
    switch (stepIndex) {
      case 0:
        return surveyData.ageGroup && surveyData.educationLevel;
      case 1:
        return surveyData.learningStyle && surveyData.preferredPace;
      case 2:
        return surveyData.primaryInterests.length > 0;
      case 3:
        return surveyData.learningChallenges.length > 0 && surveyData.accommodationsNeeded.length > 0;
      case 4:
        return surveyData.studyEnvironment && surveyData.preferredStudyTime && surveyData.sessionDuration;
      case 5:
        return surveyData.bciInterest && surveyData.adaptationComfort && surveyData.privacyConcerns;
      default:
        return false;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <div className="max-w-4xl w-full bg-gray-800 rounded-lg shadow-xl p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-4">Personalization Survey</h1>
          <p className="text-gray-400 mb-6">Help us create your perfect learning experience</p>
          
          {/* Progress Bar */}
          <div className="w-full bg-gray-700 rounded-full h-2 mb-6">
            <div 
              className="bg-gradient-to-r from-orange-500 to-red-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
            ></div>
          </div>
          
          <div className="flex justify-between text-sm text-gray-400 mb-8">
            <span>Step {currentStep + 1} of {steps.length}</span>
            <span>{Math.round(((currentStep + 1) / steps.length) * 100)}% Complete</span>
          </div>
        </div>

        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-white mb-6">{steps[currentStep].title}</h2>
          {steps[currentStep].content}
        </div>

        <div className="flex justify-between">
          <button
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            className="px-6 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            Previous
          </button>

          {currentStep === steps.length - 1 ? (
            <button
              onClick={submitSurvey}
              disabled={!isStepComplete(currentStep) || isSubmitting}
              className="px-6 py-2 bg-gradient-to-r from-orange-500 to-red-500 text-white rounded-lg hover:from-orange-600 hover:to-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {isSubmitting ? 'Submitting...' : 'Complete Survey'}
            </button>
          ) : (
            <button
              onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
              disabled={!isStepComplete(currentStep)}
              className="px-6 py-2 bg-gradient-to-r from-orange-500 to-red-500 text-white rounded-lg hover:from-orange-600 hover:to-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              Next
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default PreSurvey;