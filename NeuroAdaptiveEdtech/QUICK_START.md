# NeuroAdaptive EdTech - Quick Start Guide

## 🚀 Quick Setup & Run

### Prerequisites
- Node.js (v16+)
- Python (v3.8+)
- MongoDB (running locally or MongoDB Atlas)

### 1. Install Dependencies

**Frontend & Backend:**
```bash
# Install frontend dependencies
npm install

# Install backend dependencies
cd server
npm install
cd ..
```

**Python/BCI Dependencies:**
```bash
pip install -r requirements_fastapi.txt
```

### 2. Environment Setup

**Backend (.env in server folder):**
```env
MONGODB_URI=mongodb://localhost:27017/neuroadaptive
JWT_SECRET=your-super-secret-jwt-key-here
PORT=5001
```

**Frontend (.env.local):**
```env
VITE_API_BASE=http://localhost:5001/api
VITE_BCI_BASE=http://localhost:8000
```

### 3. Start the Complete System

**Option 1: Use the startup script (Windows)**
```bash
start_complete_system.bat
```

**Option 2: Manual startup**
```bash
# Terminal 1: Backend
cd server
npm start

# Terminal 2: BCI Service  
python fastapi_server.py

# Terminal 3: Frontend
npm run dev
```

### 4. Access the Application

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:5001
- **BCI Service:** http://localhost:8000

## 🎯 New Features

### Enhanced Authentication
- **Standalone Login Page:** `/login`
- **Standalone Signup Page:** `/signup`
- **Improved UX:** Better styling and user feedback

### Advanced Collaboration
- **Real-time Study Sessions:** Create and join collaborative learning sessions
- **AI-Generated Quizzes:** Dynamic quizzes based on any topic
- **Live Chat:** Real-time messaging during sessions
- **Gamification:** Points system and leaderboards
- **Session Sharing:** Shareable links for study sessions

### Backend Integration
- **User Management:** Complete authentication system
- **Session Management:** Persistent collaboration sessions
- **Real-time Features:** Chat and quiz synchronization

## 🔧 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user

### Collaboration
- `POST /api/collaboration/sessions` - Create session
- `POST /api/collaboration/sessions/:id/join` - Join session
- `GET /api/collaboration/sessions/:id` - Get session details
- `POST /api/collaboration/sessions/:id/answer` - Submit quiz answer
- `POST /api/collaboration/sessions/:id/chat` - Send chat message

## 🎮 How to Use

1. **Sign Up/Login:** Create an account or sign in
2. **Start Collaborating:** Go to Collaboration page
3. **Create Session:** Choose a topic and create a study session
4. **Invite Others:** Share the session link with friends
5. **Learn Together:** Answer quizzes, chat, and compete!

## 🛠️ Development Notes

- **Frontend:** React + TypeScript + Tailwind CSS
- **Backend:** Node.js + Express + MongoDB
- **BCI Service:** Python + FastAPI + PyTorch
- **Real-time:** WebSocket support ready for implementation
- **AI Integration:** Google Gemini API for quiz generation

## 🚨 Troubleshooting

**MongoDB Connection Issues:**
- Ensure MongoDB is running locally
- Check connection string in server/.env

**BCI Service Issues:**
- Install Python dependencies: `pip install -r requirements_fastapi.txt`
- The system works with simulated data if BCI model is unavailable

**Port Conflicts:**
- Frontend: Change port in vite.config.ts
- Backend: Change PORT in server/.env
- BCI: Change port in fastapi_server.py

## 🎉 Ready to Learn!

Your enhanced NeuroAdaptive EdTech platform is now ready with:
- ✅ Complete authentication system
- ✅ Real-time collaboration features  
- ✅ AI-powered quiz generation
- ✅ Gamified learning experience
- ✅ Backend integration
- ✅ Modern, responsive UI

Start the system and begin your collaborative learning journey!