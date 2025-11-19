import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';

interface SurveyData {
  userType: string;
  profession: string;
  skills: string[];
  experience: string;
  goals: string[];
  interests: string[];
  challenges: string[];
  learningStyle: string;
  timeCommitment: string;
  motivation: string;
}

const OnboardingSurvey: React.FC<{ onComplete: () => void }> = ({ onComplete }) => {
  const [currentPage, setCurrentPage] = useState(0);
  const [surveyData, setSurveyData] = useState<SurveyData>({
    userType: '',
    profession: '',
    skills: [],
    experience: '',
    goals: [],
    interests: [],
    challenges: [],
    learningStyle: '',
    timeCommitment: '',
    motivation: ''
  });
  const { updateUser } = useAuth();

  const updateData = (field: keyof SurveyData, value: any) => {
    setSurveyData(prev => ({ ...prev, [field]: value }));
  };

  const toggleArrayItem = (field: keyof SurveyData, item: string) => {
    setSurveyData(prev => ({
      ...prev,
      [field]: (prev[field] as string[]).includes(item)
        ? (prev[field] as string[]).filter(i => i !== item)
        : [...(prev[field] as string[]), item]
    }));
  };

  const handleComplete = async () => {
    try {
      // Update user profile with survey data
      updateUser({ needsSurvey: false, profile: surveyData });
      
      // Save to backend
      const response = await fetch('http://localhost:5001/api/survey/complete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(surveyData)
      });

      if (response.ok) {
        onComplete();
      }
    } catch (error) {
      console.error('Survey completion error:', error);
      onComplete(); // Continue anyway
    }
  };

  const pages = [
    // Page 1: User Type
    <div key="page1" className="survey-page">
      <div className="floating-orb orb-1"></div>
      <div className="floating-orb orb-2"></div>
      <div className="survey-content">
        <h2 className="survey-title">Who Are You?</h2>
        <p className="survey-subtitle">Help us understand your background</p>
        <div className="options-grid">
          {[
            { id: 'student', label: 'Student', icon: '🎓', desc: 'Currently studying or in education' },
            { id: 'professional', label: 'Professional', icon: '💼', desc: 'Working in your field' },
            { id: 'researcher', label: 'Researcher', icon: '🔬', desc: 'Academic or industry researcher' },
            { id: 'educator', label: 'Educator', icon: '👨‍🏫', desc: 'Teacher or trainer' },
            { id: 'entrepreneur', label: 'Entrepreneur', icon: '🚀', desc: 'Building your own venture' },
            { id: 'other', label: 'Other', icon: '🌟', desc: 'Something else entirely' }
          ].map(option => (
            <button
              key={option.id}
              onClick={() => updateData('userType', option.id)}
              className={`option-card ${surveyData.userType === option.id ? 'selected' : ''}`}
            >
              <div className="option-icon">{option.icon}</div>
              <div className="option-label">{option.label}</div>
              <div className="option-desc">{option.desc}</div>
            </button>
          ))}
        </div>
      </div>
    </div>,

    // Page 2: Profession & Skills
    <div key="page2" className="survey-page">
      <div className="floating-orb orb-3"></div>
      <div className="floating-orb orb-4"></div>
      <div className="survey-content">
        <h2 className="survey-title">Your Professional Domain</h2>
        <p className="survey-subtitle">What's your area of expertise?</p>
        
        <div className="input-section">
          <label className="input-label">Current Profession/Field</label>
          <input
            type="text"
            value={surveyData.profession}
            onChange={(e) => updateData('profession', e.target.value)}
            placeholder="e.g., Software Engineer, Data Scientist, Student..."
            className="survey-input"
          />
        </div>

        <div className="skills-section">
          <label className="input-label">Your Skills & Expertise</label>
          <div className="skills-grid">
            {[
              'Programming', 'Data Science', 'AI/ML', 'Web Development', 'Mobile Development',
              'Design', 'Marketing', 'Business', 'Research', 'Writing', 'Mathematics',
              'Physics', 'Biology', 'Chemistry', 'Engineering', 'Finance', 'Other'
            ].map(skill => (
              <button
                key={skill}
                onClick={() => toggleArrayItem('skills', skill)}
                className={`skill-tag ${surveyData.skills.includes(skill) ? 'selected' : ''}`}
              >
                {skill}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>,

    // Page 3: Experience & Goals
    <div key="page3" className="survey-page">
      <div className="floating-orb orb-5"></div>
      <div className="floating-orb orb-6"></div>
      <div className="survey-content">
        <h2 className="survey-title">Experience & Aspirations</h2>
        <p className="survey-subtitle">Where are you in your journey?</p>
        
        <div className="experience-section">
          <label className="input-label">Experience Level</label>
          <div className="options-row">
            {[
              { id: 'beginner', label: 'Beginner', desc: '0-2 years' },
              { id: 'intermediate', label: 'Intermediate', desc: '2-5 years' },
              { id: 'advanced', label: 'Advanced', desc: '5-10 years' },
              { id: 'expert', label: 'Expert', desc: '10+ years' }
            ].map(exp => (
              <button
                key={exp.id}
                onClick={() => updateData('experience', exp.id)}
                className={`experience-card ${surveyData.experience === exp.id ? 'selected' : ''}`}
              >
                <div className="exp-label">{exp.label}</div>
                <div className="exp-desc">{exp.desc}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="goals-section">
          <label className="input-label">What are your learning goals?</label>
          <div className="goals-grid">
            {[
              'Career Advancement', 'Skill Development', 'Academic Success', 'Personal Growth',
              'Research Projects', 'Startup Building', 'Teaching Others', 'Problem Solving',
              'Innovation', 'Certification', 'Networking', 'Exploration'
            ].map(goal => (
              <button
                key={goal}
                onClick={() => toggleArrayItem('goals', goal)}
                className={`goal-tag ${surveyData.goals.includes(goal) ? 'selected' : ''}`}
              >
                {goal}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>,

    // Page 4: Interests & Challenges
    <div key="page4" className="survey-page">
      <div className="floating-orb orb-7"></div>
      <div className="floating-orb orb-8"></div>
      <div className="survey-content">
        <h2 className="survey-title">Interests & Challenges</h2>
        <p className="survey-subtitle">What excites and challenges you?</p>
        
        <div className="interests-section">
          <label className="input-label">Areas of Interest</label>
          <div className="interests-grid">
            {[
              'Artificial Intelligence', 'Machine Learning', 'Data Science', 'Web Development',
              'Mobile Apps', 'Blockchain', 'Cybersecurity', 'Cloud Computing', 'IoT',
              'Robotics', 'Quantum Computing', 'Biotechnology', 'Space Technology',
              'Renewable Energy', 'Virtual Reality', 'Augmented Reality'
            ].map(interest => (
              <button
                key={interest}
                onClick={() => toggleArrayItem('interests', interest)}
                className={`interest-tag ${surveyData.interests.includes(interest) ? 'selected' : ''}`}
              >
                {interest}
              </button>
            ))}
          </div>
        </div>

        <div className="challenges-section">
          <label className="input-label">Learning Challenges</label>
          <div className="challenges-grid">
            {[
              'Time Management', 'Focus Issues', 'Information Overload', 'Lack of Motivation',
              'Technical Complexity', 'Finding Resources', 'Practical Application',
              'Staying Updated', 'Networking', 'Imposter Syndrome'
            ].map(challenge => (
              <button
                key={challenge}
                onClick={() => toggleArrayItem('challenges', challenge)}
                className={`challenge-tag ${surveyData.challenges.includes(challenge) ? 'selected' : ''}`}
              >
                {challenge}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>,

    // Page 5: Learning Preferences
    <div key="page5" className="survey-page">
      <div className="floating-orb orb-9"></div>
      <div className="floating-orb orb-10"></div>
      <div className="survey-content">
        <h2 className="survey-title">Learning Preferences</h2>
        <p className="survey-subtitle">How do you learn best?</p>
        
        <div className="learning-style-section">
          <label className="input-label">Preferred Learning Style</label>
          <div className="style-options">
            {[
              { id: 'visual', label: 'Visual', icon: '👁️', desc: 'Charts, diagrams, videos' },
              { id: 'auditory', label: 'Auditory', icon: '🎧', desc: 'Lectures, discussions, podcasts' },
              { id: 'kinesthetic', label: 'Hands-on', icon: '✋', desc: 'Practice, experiments, building' },
              { id: 'reading', label: 'Reading/Writing', icon: '📚', desc: 'Text, notes, documentation' }
            ].map(style => (
              <button
                key={style.id}
                onClick={() => updateData('learningStyle', style.id)}
                className={`style-card ${surveyData.learningStyle === style.id ? 'selected' : ''}`}
              >
                <div className="style-icon">{style.icon}</div>
                <div className="style-label">{style.label}</div>
                <div className="style-desc">{style.desc}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="time-section">
          <label className="input-label">Time Commitment</label>
          <div className="time-options">
            {[
              { id: '15min', label: '15 min/day', desc: 'Quick sessions' },
              { id: '30min', label: '30 min/day', desc: 'Regular practice' },
              { id: '1hour', label: '1 hour/day', desc: 'Focused learning' },
              { id: '2hours', label: '2+ hours/day', desc: 'Deep dive' }
            ].map(time => (
              <button
                key={time.id}
                onClick={() => updateData('timeCommitment', time.id)}
                className={`time-card ${surveyData.timeCommitment === time.id ? 'selected' : ''}`}
              >
                <div className="time-label">{time.label}</div>
                <div className="time-desc">{time.desc}</div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>,

    // Page 6: Motivation & Completion
    <div key="page6" className="survey-page">
      <div className="floating-orb orb-11"></div>
      <div className="floating-orb orb-12"></div>
      <div className="survey-content">
        <h2 className="survey-title">Final Touch</h2>
        <p className="survey-subtitle">What drives your learning journey?</p>
        
        <div className="motivation-section">
          <label className="input-label">What motivates you most?</label>
          <textarea
            value={surveyData.motivation}
            onChange={(e) => updateData('motivation', e.target.value)}
            placeholder="Share what inspires you to learn and grow..."
            className="motivation-textarea"
            rows={4}
          />
        </div>

        <div className="completion-section">
          <div className="completion-card">
            <div className="completion-icon">🎉</div>
            <h3 className="completion-title">Ready to Begin!</h3>
            <p className="completion-text">
              Your personalized learning experience awaits. We'll use this information to tailor 
              content, recommendations, and features specifically for you.
            </p>
          </div>
        </div>
      </div>
    </div>
  ];

  return (
    <div className="survey-container">
      <style>{`
        .survey-container {
          min-height: 100vh;
          background: #000;
          color: white;
          overflow: hidden;
          position: relative;
        }

        .survey-page {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 2rem;
          position: relative;
        }

        .floating-orb {
          position: absolute;
          border-radius: 50%;
          filter: blur(40px);
          animation: float 6s ease-in-out infinite;
          z-index: 0;
        }

        .orb-1 { width: 200px; height: 200px; background: linear-gradient(45deg, #ff006e, #8338ec); top: 10%; left: 10%; animation-delay: 0s; }
        .orb-2 { width: 150px; height: 150px; background: linear-gradient(45deg, #3a86ff, #06ffa5); top: 60%; right: 15%; animation-delay: 2s; }
        .orb-3 { width: 180px; height: 180px; background: linear-gradient(45deg, #ffbe0b, #fb5607); top: 20%; right: 20%; animation-delay: 1s; }
        .orb-4 { width: 120px; height: 120px; background: linear-gradient(45deg, #8338ec, #3a86ff); bottom: 20%; left: 10%; animation-delay: 3s; }
        .orb-5 { width: 160px; height: 160px; background: linear-gradient(45deg, #06ffa5, #ffbe0b); top: 15%; left: 15%; animation-delay: 0.5s; }
        .orb-6 { width: 140px; height: 140px; background: linear-gradient(45deg, #fb5607, #ff006e); bottom: 25%; right: 10%; animation-delay: 2.5s; }
        .orb-7 { width: 190px; height: 190px; background: linear-gradient(45deg, #8338ec, #06ffa5); top: 25%; right: 25%; animation-delay: 1.5s; }
        .orb-8 { width: 130px; height: 130px; background: linear-gradient(45deg, #3a86ff, #ffbe0b); bottom: 30%; left: 20%; animation-delay: 3.5s; }
        .orb-9 { width: 170px; height: 170px; background: linear-gradient(45deg, #ff006e, #06ffa5); top: 30%; left: 25%; animation-delay: 0.8s; }
        .orb-10 { width: 110px; height: 110px; background: linear-gradient(45deg, #fb5607, #3a86ff); bottom: 35%; right: 30%; animation-delay: 2.8s; }
        .orb-11 { width: 200px; height: 200px; background: linear-gradient(45deg, #ffbe0b, #8338ec); top: 20%; right: 15%; animation-delay: 1.2s; }
        .orb-12 { width: 150px; height: 150px; background: linear-gradient(45deg, #06ffa5, #ff006e); bottom: 20%; left: 15%; animation-delay: 3.2s; }

        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(180deg); }
        }

        .survey-content {
          max-width: 800px;
          width: 100%;
          z-index: 10;
          position: relative;
          backdrop-filter: blur(10px);
          background: rgba(255, 255, 255, 0.05);
          border-radius: 20px;
          padding: 3rem;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .survey-title {
          font-size: 3rem;
          font-weight: bold;
          text-align: center;
          margin-bottom: 1rem;
          background: linear-gradient(45deg, #ff006e, #8338ec, #3a86ff, #06ffa5, #ffbe0b);
          background-size: 300% 300%;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          animation: gradient 3s ease infinite;
        }

        .survey-subtitle {
          text-align: center;
          font-size: 1.2rem;
          color: rgba(255, 255, 255, 0.7);
          margin-bottom: 3rem;
        }

        @keyframes gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }

        .options-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1.5rem;
          margin-bottom: 2rem;
        }

        .option-card {
          background: rgba(255, 255, 255, 0.05);
          border: 2px solid rgba(255, 255, 255, 0.1);
          border-radius: 15px;
          padding: 2rem;
          text-align: center;
          cursor: pointer;
          transition: all 0.3s ease;
          backdrop-filter: blur(10px);
        }

        .option-card:hover {
          transform: translateY(-5px);
          border-color: rgba(255, 255, 255, 0.3);
          box-shadow: 0 10px 30px rgba(255, 255, 255, 0.1);
        }

        .option-card.selected {
          background: linear-gradient(45deg, rgba(255, 0, 110, 0.2), rgba(131, 56, 236, 0.2));
          border-color: #ff006e;
          box-shadow: 0 0 30px rgba(255, 0, 110, 0.3);
        }

        .option-icon {
          font-size: 3rem;
          margin-bottom: 1rem;
        }

        .option-label {
          font-size: 1.3rem;
          font-weight: bold;
          margin-bottom: 0.5rem;
        }

        .option-desc {
          color: rgba(255, 255, 255, 0.6);
          font-size: 0.9rem;
        }

        .input-section, .skills-section, .goals-section, .interests-section, .challenges-section, .learning-style-section, .time-section, .motivation-section {
          margin-bottom: 2rem;
        }

        .input-label {
          display: block;
          font-size: 1.1rem;
          font-weight: bold;
          margin-bottom: 1rem;
          color: rgba(255, 255, 255, 0.9);
        }

        .survey-input {
          width: 100%;
          padding: 1rem;
          background: rgba(255, 255, 255, 0.1);
          border: 2px solid rgba(255, 255, 255, 0.2);
          border-radius: 10px;
          color: white;
          font-size: 1rem;
          backdrop-filter: blur(10px);
        }

        .survey-input:focus {
          outline: none;
          border-color: #3a86ff;
          box-shadow: 0 0 20px rgba(58, 134, 255, 0.3);
        }

        .skills-grid, .goals-grid, .interests-grid, .challenges-grid {
          display: flex;
          flex-wrap: wrap;
          gap: 0.8rem;
        }

        .skill-tag, .goal-tag, .interest-tag, .challenge-tag {
          padding: 0.8rem 1.5rem;
          background: rgba(255, 255, 255, 0.1);
          border: 2px solid rgba(255, 255, 255, 0.2);
          border-radius: 25px;
          cursor: pointer;
          transition: all 0.3s ease;
          font-size: 0.9rem;
        }

        .skill-tag:hover, .goal-tag:hover, .interest-tag:hover, .challenge-tag:hover {
          transform: scale(1.05);
          border-color: rgba(255, 255, 255, 0.4);
        }

        .skill-tag.selected { background: linear-gradient(45deg, #3a86ff, #06ffa5); border-color: #3a86ff; }
        .goal-tag.selected { background: linear-gradient(45deg, #ffbe0b, #fb5607); border-color: #ffbe0b; }
        .interest-tag.selected { background: linear-gradient(45deg, #8338ec, #ff006e); border-color: #8338ec; }
        .challenge-tag.selected { background: linear-gradient(45deg, #fb5607, #ff006e); border-color: #fb5607; }

        .options-row {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 1rem;
        }

        .experience-card, .style-card, .time-card {
          background: rgba(255, 255, 255, 0.05);
          border: 2px solid rgba(255, 255, 255, 0.1);
          border-radius: 10px;
          padding: 1.5rem;
          text-align: center;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .experience-card:hover, .style-card:hover, .time-card:hover {
          transform: translateY(-3px);
          border-color: rgba(255, 255, 255, 0.3);
        }

        .experience-card.selected, .style-card.selected, .time-card.selected {
          background: linear-gradient(45deg, rgba(58, 134, 255, 0.2), rgba(6, 255, 165, 0.2));
          border-color: #3a86ff;
          box-shadow: 0 0 20px rgba(58, 134, 255, 0.3);
        }

        .style-options {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1rem;
        }

        .style-icon {
          font-size: 2rem;
          margin-bottom: 0.5rem;
        }

        .motivation-textarea {
          width: 100%;
          padding: 1rem;
          background: rgba(255, 255, 255, 0.1);
          border: 2px solid rgba(255, 255, 255, 0.2);
          border-radius: 10px;
          color: white;
          font-size: 1rem;
          resize: vertical;
          backdrop-filter: blur(10px);
        }

        .motivation-textarea:focus {
          outline: none;
          border-color: #8338ec;
          box-shadow: 0 0 20px rgba(131, 56, 236, 0.3);
        }

        .completion-card {
          text-align: center;
          background: linear-gradient(45deg, rgba(255, 0, 110, 0.1), rgba(131, 56, 236, 0.1));
          border: 2px solid rgba(255, 0, 110, 0.3);
          border-radius: 20px;
          padding: 2rem;
          margin-bottom: 2rem;
        }

        .completion-icon {
          font-size: 4rem;
          margin-bottom: 1rem;
        }

        .completion-title {
          font-size: 2rem;
          font-weight: bold;
          margin-bottom: 1rem;
          color: #ff006e;
        }

        .completion-text {
          color: rgba(255, 255, 255, 0.8);
          line-height: 1.6;
        }

        .navigation {
          position: fixed;
          bottom: 2rem;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          gap: 1rem;
          z-index: 20;
        }

        .nav-btn {
          padding: 1rem 2rem;
          background: rgba(255, 255, 255, 0.1);
          border: 2px solid rgba(255, 255, 255, 0.2);
          border-radius: 50px;
          color: white;
          cursor: pointer;
          transition: all 0.3s ease;
          backdrop-filter: blur(10px);
          font-weight: bold;
        }

        .nav-btn:hover {
          background: rgba(255, 255, 255, 0.2);
          transform: scale(1.05);
        }

        .nav-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .nav-btn.primary {
          background: linear-gradient(45deg, #ff006e, #8338ec);
          border-color: #ff006e;
        }

        .nav-btn.primary:hover {
          box-shadow: 0 0 30px rgba(255, 0, 110, 0.4);
        }

        .progress-bar {
          position: fixed;
          top: 0;
          left: 0;
          height: 4px;
          background: linear-gradient(90deg, #ff006e, #8338ec, #3a86ff, #06ffa5, #ffbe0b);
          transition: width 0.3s ease;
          z-index: 30;
        }
      `}</style>

      <div className="progress-bar" style={{ width: `${((currentPage + 1) / pages.length) * 100}%` }}></div>
      
      {pages[currentPage]}

      <div className="navigation">
        {currentPage > 0 && (
          <button
            onClick={() => setCurrentPage(prev => prev - 1)}
            className="nav-btn"
          >
            ← Previous
          </button>
        )}
        
        {currentPage < pages.length - 1 ? (
          <button
            onClick={() => setCurrentPage(prev => prev + 1)}
            className="nav-btn primary"
            disabled={
              (currentPage === 0 && !surveyData.userType) ||
              (currentPage === 1 && !surveyData.profession) ||
              (currentPage === 2 && !surveyData.experience) ||
              (currentPage === 3 && surveyData.interests.length === 0) ||
              (currentPage === 4 && !surveyData.learningStyle)
            }
          >
            Next →
          </button>
        ) : (
          <button
            onClick={handleComplete}
            className="nav-btn primary"
            disabled={!surveyData.motivation.trim()}
          >
            Complete Survey 🎉
          </button>
        )}
      </div>
    </div>
  );
};

export default OnboardingSurvey;