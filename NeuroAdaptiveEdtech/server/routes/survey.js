import express from 'express';
import jwt from 'jsonwebtoken';
import User from '../models/User.js';
import SurveyResponse from '../models/Survey.js';

const router = express.Router();

// Middleware to verify JWT token
const authenticateToken = (req, res, next) => {
  const token = req.header('Authorization')?.replace('Bearer ', '');
  if (!token) {
    return res.status(401).json({ message: 'Access denied. No token provided.' });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.userId = decoded.userId;
    next();
  } catch (error) {
    res.status(400).json({ message: 'Invalid token.' });
  }
};

// Submit survey response
router.post('/submit', authenticateToken, async (req, res) => {
  try {
    const { responses } = req.body;
    const userId = req.userId;

    // Generate personalized recommendations based on responses
    const recommendations = generateRecommendations(responses);

    // Save survey response
    const surveyResponse = new SurveyResponse({
      userId,
      responses,
      recommendations
    });

    await surveyResponse.save();

    // Update user profile with survey data
    const user = await User.findById(userId);
    if (user) {
      user.profile = {
        ageGroup: responses.ageGroup,
        learningStyle: responses.learningStyle,
        interests: responses.primaryInterests,
        challenges: responses.learningChallenges,
        goals: responses.careerGoals,
        preferredContentTypes: responses.preferredContentTypes,
        studyHabits: {
          preferredTime: responses.preferredStudyTime,
          sessionDuration: responses.sessionDuration,
          breakFrequency: responses.breakFrequency
        }
      };

      user.bciPreferences = {
        enableBCI: responses.bciInterest === 'very-interested' || responses.bciInterest === 'somewhat-interested',
        adaptationLevel: responses.adaptationComfort,
        interventionTypes: recommendations.interventionTypes
      };

      await user.save();
    }

    res.json({
      message: 'Survey submitted successfully',
      recommendations,
      profile: user.profile
    });
  } catch (error) {
    console.error('Survey submission error:', error);
    res.status(500).json({ message: 'Server error during survey submission' });
  }
});

// Complete comprehensive onboarding survey
router.post('/complete', authenticateToken, async (req, res) => {
  try {
    const {
      userType,
      profession,
      skills,
      experience,
      goals,
      interests,
      challenges,
      learningStyle,
      timeCommitment,
      motivation
    } = req.body;

    const user = await User.findByIdAndUpdate(
      req.userId,
      {
        'profile.userType': userType,
        'profile.profession': profession,
        'profile.skills': skills,
        'profile.experience': experience,
        'profile.goals': goals,
        'profile.interests': interests,
        'profile.challenges': challenges,
        'profile.learningStyle': learningStyle,
        'profile.timeCommitment': timeCommitment,
        'profile.motivation': motivation,
        'profile.ageGroup': '18-22',
        'profile.preferredContentTypes': ['interactive', 'videos'],
        'profile.studyHabits': {
          preferredTime: 'morning',
          sessionDuration: timeCommitment === '15min' ? '15-30min' : timeCommitment === '30min' ? '30-60min' : '60-90min',
          breakFrequency: 'every-30min'
        }
      },
      { new: true }
    );

    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    res.json({
      message: 'Survey completed successfully',
      user: {
        id: user._id,
        email: user.email,
        name: user.name,
        needsSurvey: false,
        profile: user.profile
      }
    });
  } catch (error) {
    console.error('Survey completion error:', error);
    res.status(500).json({ message: 'Server error completing survey' });
  }
});

// Get user's survey response
router.get('/response', authenticateToken, async (req, res) => {
  try {
    const surveyResponse = await SurveyResponse.findOne({ userId: req.userId }).sort({ createdAt: -1 });
    
    if (!surveyResponse) {
      return res.status(404).json({ message: 'No survey response found' });
    }

    res.json(surveyResponse);
  } catch (error) {
    console.error('Get survey error:', error);
    res.status(500).json({ message: 'Server error retrieving survey' });
  }
});

// Generate personalized recommendations based on survey responses
function generateRecommendations(responses) {
  const recommendations = {
    contentTypes: [],
    studySchedule: {},
    adaptiveFeatures: [],
    interventionTypes: []
  };

  // Content type recommendations based on learning style and preferences
  if (responses.learningStyle === 'visual' || responses.preferredContentTypes.includes('videos')) {
    recommendations.contentTypes.push('interactive-videos', 'infographics', 'mind-maps');
  }
  if (responses.learningStyle === 'auditory' || responses.preferredContentTypes.includes('audio-podcasts')) {
    recommendations.contentTypes.push('audio-lectures', 'discussion-forums', 'voice-notes');
  }
  if (responses.learningStyle === 'kinesthetic' || responses.preferredContentTypes.includes('interactive-simulations')) {
    recommendations.contentTypes.push('hands-on-labs', 'vr-experiences', 'interactive-simulations');
  }

  // Study schedule recommendations
  recommendations.studySchedule = {
    sessionLength: responses.sessionDuration,
    breakInterval: responses.breakFrequency,
    dailyGoal: getDailyGoalRecommendation(responses.sessionDuration, responses.ageGroup)
  };

  // Adaptive features based on challenges and preferences
  if (responses.learningChallenges.includes('ADHD')) {
    recommendations.adaptiveFeatures.push('focus-timers', 'gamification', 'micro-learning');
    recommendations.interventionTypes.push('visual', 'audio');
  }
  if (responses.learningChallenges.includes('Dyslexia')) {
    recommendations.adaptiveFeatures.push('text-to-speech', 'dyslexia-friendly-fonts', 'visual-aids');
    recommendations.interventionTypes.push('audio', 'visual');
  }
  if (responses.learningChallenges.includes('Anxiety')) {
    recommendations.adaptiveFeatures.push('stress-monitoring', 'breathing-exercises', 'calm-environments');
    recommendations.interventionTypes.push('content-change', 'audio');
  }

  // BCI-specific recommendations
  if (responses.bciInterest === 'very-interested') {
    recommendations.adaptiveFeatures.push('real-time-attention-monitoring', 'cognitive-load-optimization');
    recommendations.interventionTypes.push('visual', 'audio', 'content-change');
  }

  // Technology comfort level adaptations
  if (responses.deviceComfort === 'beginner') {
    recommendations.adaptiveFeatures.push('simplified-interface', 'guided-tutorials');
  } else if (responses.deviceComfort === 'advanced' || responses.deviceComfort === 'expert') {
    recommendations.adaptiveFeatures.push('advanced-analytics', 'customizable-dashboard', 'api-access');
  }

  return recommendations;
}

function getDailyGoalRecommendation(sessionDuration, ageGroup) {
  const sessionMinutes = {
    '15-30min': 22.5,
    '30-45min': 37.5,
    '45-60min': 52.5,
    '60-90min': 75,
    '90min+': 120
  };

  const baseMinutes = sessionMinutes[sessionDuration] || 45;
  
  // Adjust based on age group
  const ageMultiplier = {
    '12-18': 1.0,
    '18-22': 1.2,
    '22+': 1.5
  };

  const recommendedDaily = Math.round(baseMinutes * (ageMultiplier[ageGroup] || 1.0));
  return `${recommendedDaily} minutes per day`;
}

export default router;