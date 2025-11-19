// Content delivery service that adapts based on user profile and BCI state

export class ContentService {
  constructor(userProfile, bciState = null) {
    this.userProfile = userProfile;
    this.bciState = bciState;
  }

  // Get personalized content based on user profile
  getPersonalizedContent(subject, difficulty = 'intermediate') {
    const baseContent = this.getBaseContent(subject, difficulty);
    
    if (!this.userProfile) {
      return baseContent;
    }

    // Adapt content based on learning style
    const adaptedContent = this.adaptForLearningStyle(baseContent);
    
    // Adapt for challenges/accommodations
    const accommodatedContent = this.adaptForChallenges(adaptedContent);
    
    // Apply BCI-based adaptations if available
    const bciAdaptedContent = this.bciState ? 
      this.adaptForBCIState(accommodatedContent) : accommodatedContent;

    return bciAdaptedContent;
  }

  getBaseContent(subject, difficulty) {
    const contentLibrary = {
      mathematics: {
        beginner: {
          title: "Introduction to Basic Math",
          description: "Learn fundamental mathematical concepts",
          modules: [
            { type: 'video', title: 'Numbers and Operations', duration: '15 min' },
            { type: 'interactive', title: 'Practice Problems', duration: '20 min' },
            { type: 'quiz', title: 'Assessment', duration: '10 min' }
          ]
        },
        intermediate: {
          title: "Algebra Fundamentals",
          description: "Master algebraic thinking and problem solving",
          modules: [
            { type: 'video', title: 'Linear Equations', duration: '25 min' },
            { type: 'interactive', title: 'Graphing Practice', duration: '30 min' },
            { type: 'text', title: 'Theory Review', duration: '15 min' },
            { type: 'quiz', title: 'Chapter Test', duration: '20 min' }
          ]
        },
        advanced: {
          title: "Calculus Concepts",
          description: "Advanced mathematical analysis",
          modules: [
            { type: 'video', title: 'Derivatives', duration: '35 min' },
            { type: 'interactive', title: 'Function Analysis', duration: '40 min' },
            { type: 'text', title: 'Theoretical Foundations', duration: '25 min' }
          ]
        }
      },
      science: {
        beginner: {
          title: "Science Basics",
          description: "Explore the fundamentals of scientific thinking",
          modules: [
            { type: 'video', title: 'Scientific Method', duration: '20 min' },
            { type: 'interactive', title: 'Virtual Lab', duration: '25 min' },
            { type: 'quiz', title: 'Knowledge Check', duration: '10 min' }
          ]
        },
        intermediate: {
          title: "Physics Principles",
          description: "Understanding motion, energy, and forces",
          modules: [
            { type: 'video', title: 'Newton\'s Laws', duration: '30 min' },
            { type: 'interactive', title: 'Physics Simulator', duration: '35 min' },
            { type: 'text', title: 'Problem Solving Guide', duration: '20 min' }
          ]
        }
      },
      technology: {
        beginner: {
          title: "Computer Basics",
          description: "Introduction to computing and digital literacy",
          modules: [
            { type: 'video', title: 'How Computers Work', duration: '18 min' },
            { type: 'interactive', title: 'Typing Practice', duration: '15 min' },
            { type: 'text', title: 'Digital Safety', duration: '12 min' }
          ]
        },
        intermediate: {
          title: "Programming Fundamentals",
          description: "Learn to code with Python",
          modules: [
            { type: 'video', title: 'Python Basics', duration: '40 min' },
            { type: 'interactive', title: 'Code Playground', duration: '45 min' },
            { type: 'text', title: 'Best Practices', duration: '20 min' }
          ]
        }
      }
    };

    return contentLibrary[subject]?.[difficulty] || contentLibrary.mathematics.intermediate;
  }

  adaptForLearningStyle(content) {
    if (!this.userProfile?.learningStyle) return content;

    const adaptedContent = { ...content };
    
    switch (this.userProfile.learningStyle) {
      case 'visual':
        adaptedContent.modules = content.modules.map(module => {
          if (module.type === 'text') {
            return { ...module, enhancements: ['infographics', 'diagrams', 'visual-aids'] };
          }
          return module;
        });
        adaptedContent.modules.unshift({
          type: 'interactive',
          title: 'Visual Overview',
          duration: '10 min',
          description: 'Interactive visual summary'
        });
        break;

      case 'auditory':
        adaptedContent.modules = content.modules.map(module => {
          if (module.type === 'text') {
            return { ...module, enhancements: ['text-to-speech', 'audio-narration'] };
          }
          return module;
        });
        adaptedContent.modules.push({
          type: 'audio',
          title: 'Discussion Podcast',
          duration: '15 min',
          description: 'Expert discussion on key concepts'
        });
        break;

      case 'kinesthetic':
        adaptedContent.modules.push({
          type: 'interactive',
          title: 'Hands-on Lab',
          duration: '25 min',
          description: 'Interactive simulation and practice'
        });
        break;
    }

    return adaptedContent;
  }

  adaptForChallenges(content) {
    if (!this.userProfile?.challenges) return content;

    const adaptedContent = { ...content };
    const challenges = this.userProfile.challenges;

    if (challenges.includes('ADHD')) {
      adaptedContent.modules = content.modules.map(module => ({
        ...module,
        duration: this.reduceDuration(module.duration),
        enhancements: [...(module.enhancements || []), 'focus-timer', 'progress-tracker']
      }));
      
      adaptedContent.studyTips = [
        'Take breaks every 15-20 minutes',
        'Use the focus timer to stay on track'
      ];
    }

    if (challenges.includes('Dyslexia')) {
      adaptedContent.modules = content.modules.map(module => ({
        ...module,
        enhancements: [...(module.enhancements || []), 'dyslexia-font', 'text-to-speech']
      }));
    }

    return adaptedContent;
  }

  adaptForBCIState(content) {
    if (!this.bciState) return content;

    const adaptedContent = { ...content };
    const { currentState, confidence } = this.bciState;

    switch (currentState) {
      case 'Focused':
        if (confidence > 0.7) {
          adaptedContent.suggestion = "Great focus! You're ready for challenging material.";
        }
        break;

      case 'Unfocused':
        adaptedContent.modules = content.modules.map(module => ({
          ...module,
          duration: this.reduceDuration(module.duration),
          enhancements: [...(module.enhancements || []), 'attention-grabber']
        }));
        adaptedContent.suggestion = "Let's try some interactive content to regain focus.";
        break;

      case 'Drowsy':
        if (confidence > 0.6) {
          adaptedContent.suggestion = "You seem tired. Consider taking a short break.";
        }
        break;
    }

    return adaptedContent;
  }

  reduceDuration(duration) {
    const minutes = parseInt(duration);
    const reduced = Math.max(5, Math.floor(minutes * 0.7));
    return `${reduced} min`;
  }
}

export const createContentService = (userProfile, bciState = null) => {
  return new ContentService(userProfile, bciState);
};