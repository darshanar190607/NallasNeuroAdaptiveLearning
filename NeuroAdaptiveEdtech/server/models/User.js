import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';

const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true
  },
  password: {
    type: String,
    required: true,
    minlength: 6
  },
  name: {
    type: String,
    required: true
  },
  profile: {
    userType: {
      type: String,
      enum: ['student', 'professional', 'researcher', 'educator', 'entrepreneur', 'other']
    },
    profession: {
      type: String
    },
    skills: [{
      type: String
    }],
    experience: {
      type: String,
      enum: ['beginner', 'intermediate', 'advanced', 'expert']
    },
    timeCommitment: {
      type: String,
      enum: ['15min', '30min', '1hour', '2hours']
    },
    motivation: {
      type: String
    },
    ageGroup: {
      type: String,
      enum: ['12-18', '18-22', '22+'],
      default: '18-22'
    },
    learningStyle: {
      type: String,
      enum: ['visual', 'auditory', 'kinesthetic', 'reading', 'mixed']
    },
    interests: [{
      type: String
    }],
    challenges: [{
      type: String
    }],
    goals: [{
      type: String
    }],
    preferredContentTypes: [{
      type: String,
      enum: ['videos', 'interactive', 'text', 'audio', 'vr', 'ar']
    }],
    studyHabits: {
      preferredTime: {
        type: String,
        enum: ['morning', 'afternoon', 'evening', 'night']
      },
      sessionDuration: {
        type: String,
        enum: ['15-30min', '30-60min', '60-90min', '90min+']
      },
      breakFrequency: {
        type: String,
        enum: ['every-15min', 'every-30min', 'every-hour', 'as-needed']
      }
    }
  },
  bciPreferences: {
    enableBCI: {
      type: Boolean,
      default: false
    },
    adaptationLevel: {
      type: String,
      enum: ['minimal', 'moderate', 'aggressive'],
      default: 'moderate'
    },
    interventionTypes: [{
      type: String,
      enum: ['visual', 'audio', 'haptic', 'content-change']
    }]
  },
  learningProgress: {
    completedCourses: [{
      courseId: String,
      completionDate: Date,
      score: Number
    }],
    currentCourses: [{
      courseId: String,
      progress: Number,
      lastAccessed: Date
    }],
    totalStudyTime: {
      type: Number,
      default: 0
    },
    averageAttentionScore: {
      type: Number,
      default: 0
    }
  }
}, {
  timestamps: true
});

// Hash password before saving
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  this.password = await bcrypt.hash(this.password, 12);
  next();
});

// Compare password method
userSchema.methods.comparePassword = async function(candidatePassword) {
  return await bcrypt.compare(candidatePassword, this.password);
};

export default mongoose.model('User', userSchema);