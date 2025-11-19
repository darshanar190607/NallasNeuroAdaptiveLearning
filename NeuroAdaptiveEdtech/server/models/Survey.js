import mongoose from 'mongoose';

const surveyResponseSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  responses: {
    // Basic Information
    ageGroup: {
      type: String,
      enum: ['12-18', '18-22', '22+'],
      required: true
    },
    educationLevel: {
      type: String,
      enum: ['middle-school', 'high-school', 'undergraduate', 'graduate', 'professional'],
      required: true
    },
    
    // Learning Preferences
    learningStyle: {
      type: String,
      enum: ['visual', 'auditory', 'kinesthetic', 'reading-writing', 'mixed'],
      required: true
    },
    preferredPace: {
      type: String,
      enum: ['slow', 'moderate', 'fast', 'adaptive'],
      required: true
    },
    
    // Interests and Goals
    primaryInterests: [{
      type: String,
      enum: ['STEM', 'Arts', 'Business', 'Languages', 'Sports', 'Technology', 'Science', 'Mathematics', 'History', 'Literature']
    }],
    careerGoals: [{
      type: String
    }],
    
    // Challenges and Accommodations
    learningChallenges: [{
      type: String,
      enum: ['ADHD', 'Dyslexia', 'Anxiety', 'Focus Issues', 'Memory Issues', 'Processing Speed', 'None']
    }],
    accommodationsNeeded: [{
      type: String,
      enum: ['Extended Time', 'Frequent Breaks', 'Visual Aids', 'Audio Support', 'Simplified Instructions', 'None']
    }],
    
    // Technology Preferences
    deviceComfort: {
      type: String,
      enum: ['beginner', 'intermediate', 'advanced', 'expert'],
      required: true
    },
    preferredContentTypes: [{
      type: String,
      enum: ['videos', 'interactive-simulations', 'text-articles', 'audio-podcasts', 'vr-experiences', 'ar-overlays', 'games']
    }],
    
    // Study Habits
    studyEnvironment: {
      type: String,
      enum: ['quiet-room', 'background-music', 'nature-sounds', 'collaborative-space'],
      required: true
    },
    preferredStudyTime: {
      type: String,
      enum: ['early-morning', 'morning', 'afternoon', 'evening', 'night'],
      required: true
    },
    sessionDuration: {
      type: String,
      enum: ['15-30min', '30-45min', '45-60min', '60-90min', '90min+'],
      required: true
    },
    breakFrequency: {
      type: String,
      enum: ['every-15min', 'every-30min', 'every-45min', 'every-hour', 'as-needed'],
      required: true
    },
    
    // BCI and Adaptive Features
    bciInterest: {
      type: String,
      enum: ['very-interested', 'somewhat-interested', 'neutral', 'not-interested'],
      required: true
    },
    adaptationComfort: {
      type: String,
      enum: ['minimal', 'moderate', 'aggressive', 'full-automation'],
      required: true
    },
    privacyConcerns: {
      type: String,
      enum: ['none', 'minimal', 'moderate', 'high'],
      required: true
    }
  },
  
  // Generated Recommendations
  recommendations: {
    contentTypes: [String],
    studySchedule: {
      sessionLength: String,
      breakInterval: String,
      dailyGoal: String
    },
    adaptiveFeatures: [String],
    interventionTypes: [String]
  },
  
  completedAt: {
    type: Date,
    default: Date.now
  }
}, {
  timestamps: true
});

export default mongoose.model('SurveyResponse', surveyResponseSchema);