import express from 'express';
import jwt from 'jsonwebtoken';
import CollaborationSession from '../models/CollaborationSession.js';
import User from '../models/User.js';

const router = express.Router();

// Middleware to verify JWT token
const authenticateToken = (req, res, next) => {
  const token = req.header('Authorization')?.replace('Bearer ', '');
  if (!token) {
    return res.status(401).json({ message: 'No token provided' });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.userId = decoded.userId;
    next();
  } catch (error) {
    res.status(401).json({ message: 'Invalid token' });
  }
};

// Create a new collaboration session
router.post('/sessions', authenticateToken, async (req, res) => {
  try {
    const { topic, quiz } = req.body;
    const sessionId = Math.random().toString(36).substring(2, 10);
    
    const user = await User.findById(req.userId);
    
    const session = new CollaborationSession({
      sessionId,
      topic,
      creator: req.userId,
      quiz,
      participants: [{
        user: req.userId,
        points: 0,
        studyTime: 0
      }]
    });

    await session.save();
    await session.populate('participants.user', 'name email');

    res.status(201).json({
      message: 'Session created successfully',
      session: {
        id: session._id,
        sessionId: session.sessionId,
        topic: session.topic,
        participants: session.participants,
        quiz: session.quiz,
        status: session.status
      }
    });
  } catch (error) {
    console.error('Create session error:', error);
    res.status(500).json({ message: 'Server error creating session' });
  }
});

// Join a collaboration session
router.post('/sessions/:sessionId/join', authenticateToken, async (req, res) => {
  try {
    const { sessionId } = req.params;
    const session = await CollaborationSession.findOne({ sessionId });
    
    if (!session) {
      return res.status(404).json({ message: 'Session not found' });
    }

    // Check if user is already in session
    const existingParticipant = session.participants.find(p => p.user.toString() === req.userId);
    if (existingParticipant) {
      await session.populate('participants.user', 'name email');
      return res.json({
        message: 'Already in session',
        session: {
          id: session._id,
          sessionId: session.sessionId,
          topic: session.topic,
          participants: session.participants,
          quiz: session.quiz,
          status: session.status,
          currentQuestionIndex: session.currentQuestionIndex,
          chatMessages: session.chatMessages
        }
      });
    }

    // Add user to session
    session.participants.push({
      user: req.userId,
      points: 0,
      studyTime: 0
    });

    await session.save();
    await session.populate('participants.user', 'name email');

    res.json({
      message: 'Joined session successfully',
      session: {
        id: session._id,
        sessionId: session.sessionId,
        topic: session.topic,
        participants: session.participants,
        quiz: session.quiz,
        status: session.status,
        currentQuestionIndex: session.currentQuestionIndex,
        chatMessages: session.chatMessages
      }
    });
  } catch (error) {
    console.error('Join session error:', error);
    res.status(500).json({ message: 'Server error joining session' });
  }
});

// Get session details
router.get('/sessions/:sessionId', authenticateToken, async (req, res) => {
  try {
    const { sessionId } = req.params;
    const session = await CollaborationSession.findOne({ sessionId })
      .populate('participants.user', 'name email');
    
    if (!session) {
      return res.status(404).json({ message: 'Session not found' });
    }

    res.json({
      session: {
        id: session._id,
        sessionId: session.sessionId,
        topic: session.topic,
        participants: session.participants,
        quiz: session.quiz,
        status: session.status,
        currentQuestionIndex: session.currentQuestionIndex,
        chatMessages: session.chatMessages,
        answers: session.answers
      }
    });
  } catch (error) {
    console.error('Get session error:', error);
    res.status(500).json({ message: 'Server error getting session' });
  }
});

// Submit answer to quiz question
router.post('/sessions/:sessionId/answer', authenticateToken, async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { questionIndex, answerIndex } = req.body;
    
    const session = await CollaborationSession.findOne({ sessionId });
    if (!session) {
      return res.status(404).json({ message: 'Session not found' });
    }

    // Check if user already answered this question
    const existingAnswer = session.answers.find(
      a => a.questionIndex === questionIndex && a.userId.toString() === req.userId
    );
    
    if (existingAnswer) {
      return res.status(400).json({ message: 'Already answered this question' });
    }

    // Add answer
    session.answers.push({
      questionIndex,
      userId: req.userId,
      answerIndex
    });

    // Update points if correct
    const question = session.quiz[questionIndex];
    if (question && question.correct === answerIndex) {
      const participant = session.participants.find(p => p.user.toString() === req.userId);
      if (participant) {
        participant.points += 100;
      }
    }

    await session.save();
    await session.populate('participants.user', 'name email');

    res.json({
      message: 'Answer submitted successfully',
      isCorrect: question && question.correct === answerIndex,
      session: {
        participants: session.participants,
        answers: session.answers
      }
    });
  } catch (error) {
    console.error('Submit answer error:', error);
    res.status(500).json({ message: 'Server error submitting answer' });
  }
});

// Send chat message
router.post('/sessions/:sessionId/chat', authenticateToken, async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { message } = req.body;
    
    const session = await CollaborationSession.findOne({ sessionId });
    if (!session) {
      return res.status(404).json({ message: 'Session not found' });
    }

    const user = await User.findById(req.userId);
    
    session.chatMessages.push({
      sender: req.userId,
      senderName: user.name,
      message
    });

    await session.save();

    res.json({
      message: 'Message sent successfully',
      chatMessage: {
        sender: req.userId,
        senderName: user.name,
        message,
        timestamp: new Date()
      }
    });
  } catch (error) {
    console.error('Send message error:', error);
    res.status(500).json({ message: 'Server error sending message' });
  }
});

// Update study time
router.post('/sessions/:sessionId/study-time', authenticateToken, async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { studyTime } = req.body;
    
    const session = await CollaborationSession.findOne({ sessionId });
    if (!session) {
      return res.status(404).json({ message: 'Session not found' });
    }

    const participant = session.participants.find(p => p.user.toString() === req.userId);
    if (participant) {
      participant.studyTime = studyTime;
      await session.save();
    }

    res.json({ message: 'Study time updated' });
  } catch (error) {
    console.error('Update study time error:', error);
    res.status(500).json({ message: 'Server error updating study time' });
  }
});

export default router;