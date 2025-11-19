# NeuroAdaptive EdTech Platform - Setup Instructions

## Overview
This is a complete neuroadaptive learning platform that integrates:
- User authentication and personalized surveys
- Brain-Computer Interface (BCI) simulation
- Adaptive content delivery based on user profiles
- Real-time attention monitoring
- MongoDB Atlas cloud database

## Prerequisites
- Node.js (v16 or higher)
- Python 3.8+
- Git

## Installation Steps

### 1. Install Frontend Dependencies
```bash
npm install
```

### 2. Install Backend Dependencies
```bash
cd server
npm install
cd ..
```

### 3. Install Python Dependencies for BCI
```bash
pip install fastapi uvicorn torch torchvision numpy scikit-learn joblib python-multipart
```

### 4. Environment Configuration
The system is pre-configured with MongoDB Atlas connection:
- Database: `mongodb+srv://gokulnaathn64_db_user:YKKRQ1D8OkoMSVoE@cluster0.inxcm34.mongodb.net/neuroadaptive_learning`
- All environment variables are set in `server/.env`

### 5. Start the Complete System
Run the startup script:
```bash
start_complete_system.bat
```

This will start:
- Backend API Server (Port 5000)
- BCI FastAPI Server (Port 8000)
- Frontend Development Server (Port 5173)

## System Architecture

### Authentication Flow
1. Users register/login through the frontend
2. JWT tokens are stored in localStorage
3. Protected routes require authentication
4. New users are prompted to complete a personalization survey

### Survey System
- Multi-step survey collects:
  - Age group and education level
  - Learning preferences and style
  - Interests and career goals
  - Learning challenges (ADHD, Dyslexia, etc.)
  - Study habits and preferences
  - BCI interest and privacy concerns
- Generates personalized recommendations
- Updates user profile in MongoDB

### BCI Integration
- FastAPI server provides attention state prediction
- Simulated EEG data for testing
- Real-time monitoring during learning sessions
- Adaptive content based on attention state

### Content Adaptation
- Dynamic content delivery based on user profile
- Learning style adaptations (visual, auditory, kinesthetic)
- Challenge accommodations (ADHD timers, dyslexia fonts)
- BCI-driven real-time adaptations

## Usage Instructions

### 1. First Time Setup
1. Open http://localhost:5173
2. Click "Sign In" and create a new account
3. Complete the personalization survey
4. Access your adaptive learning dashboard

### 2. Learning Experience
1. Choose a subject (Mathematics, Science, Technology)
2. Start a learning session
3. BCI monitoring begins (if enabled)
4. Content adapts based on your profile and attention state
5. Progress through personalized modules

### 3. BCI Features
- Enable BCI in survey for attention monitoring
- Real-time feedback on focus levels
- Adaptive interventions based on attention state
- Simulated data for demonstration

## Database Schema

### Users Collection
```javascript
{
  email: String,
  password: String (hashed),
  name: String,
  profile: {
    ageGroup: String,
    learningStyle: String,
    interests: [String],
    challenges: [String],
    studyHabits: Object
  },
  bciPreferences: {
    enableBCI: Boolean,
    adaptationLevel: String
  },
  learningProgress: Object
}
```

### Survey Responses Collection
```javascript
{
  userId: ObjectId,
  responses: Object,
  recommendations: Object,
  completedAt: Date
}
```

## API Endpoints

### Authentication
- POST `/api/auth/register` - User registration
- POST `/api/auth/login` - User login
- GET `/api/auth/me` - Get current user

### Survey
- POST `/api/survey/submit` - Submit survey responses
- GET `/api/survey/response` - Get user's survey

### BCI
- POST `/api/bci/predict` - Get attention prediction
- POST `/api/bci/simulate` - Get simulated data
- GET `/api/bci/health` - Check BCI service health

## Features Implemented

### ✅ Core Features
- [x] User authentication (register/login)
- [x] MongoDB Atlas integration
- [x] Comprehensive personalization survey
- [x] BCI simulation and integration
- [x] Adaptive content delivery
- [x] Real-time attention monitoring
- [x] Learning style adaptations
- [x] Challenge accommodations (ADHD, Dyslexia)

### ✅ Advanced Features
- [x] JWT-based authentication
- [x] Profile-based content personalization
- [x] BCI state-driven adaptations
- [x] Multi-step survey with validation
- [x] Responsive UI with Tailwind CSS
- [x] Real-time BCI status monitoring

## Troubleshooting

### Common Issues
1. **MongoDB Connection**: Ensure internet connection for Atlas access
2. **BCI Server**: Python dependencies must be installed
3. **Port Conflicts**: Ensure ports 5000, 5173, 8000 are available
4. **CORS Issues**: Backend configured for localhost development

### Development Mode
- Frontend: `npm run dev`
- Backend: `cd server && npm start`
- BCI: `python fastapi_server.py`

## Next Steps for Enhancement
1. Implement real BCI hardware integration
2. Add more learning content and subjects
3. Implement collaborative learning features
4. Add progress analytics and reporting
5. Integrate AR/VR components
6. Add quantum simulation features

## Support
For issues or questions, check the console logs in each service window for debugging information.