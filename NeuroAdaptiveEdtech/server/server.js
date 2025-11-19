import 'dotenv/config';
import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import interactionRoutes from './routes/interactions.js';
import authRoutes from './routes/auth.js';
import surveyRoutes from './routes/survey.js';
import collaborationRoutes from './routes/collaboration.js';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import axios from 'axios';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5001;
console.log(`Attempting to start server on port ${PORT}...`);
const BCI_SERVICE_URL = process.env.BCI_SERVICE_URL || 'http://localhost:8000';

// Middleware
app.use(cors());
app.use(express.json());

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI)
.then(() => console.log('Connected to MongoDB'))
.catch(err => console.error('MongoDB connection error:', err));

// Routes
app.use('/api/interactions', interactionRoutes);
app.use('/api/auth', authRoutes);
app.use('/api/survey', surveyRoutes);
app.use('/api/collaboration', collaborationRoutes);

// BCI Routes
app.post('/api/bci/predict', async (req, res) => {
  try {
    // Forward request to FastAPI BCI service
    const response = await axios.post('http://localhost:8000/predict', req.body, {
      timeout: 15000, // 15 second timeout
      headers: {
        'Content-Type': 'application/json'
      }
    });
    res.json(response.data);
  } catch (error) {
    console.error('BCI service error:', error.message);
    
    // Try simulation endpoint as fallback
    try {
      const fallbackResponse = await axios.post('http://localhost:8000/simulate', {}, {
        timeout: 5000
      });
      res.json({
        ...fallbackResponse.data,
        simulated: true,
        message: 'Using simulated data - BCI model unavailable'
      });
    } catch (fallbackError) {
      res.status(500).json({
        error: 'BCI service unavailable',
        details: error.message,
        fallback_error: fallbackError.message
      });
    }
  }
});

// BCI Health Check
app.get('/api/bci/health', async (req, res) => {
  try {
    const response = await axios.get('http://localhost:8000/health', {
      timeout: 5000
    });
    res.json(response.data);
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message
    });
  }
});

// BCI Simulation endpoint
app.post('/api/bci/simulate', async (req, res) => {
  try {
    const response = await axios.post('http://localhost:8000/simulate', {}, {
      timeout: 5000
    });
    res.json(response.data);
  } catch (error) {
    res.status(500).json({
      error: 'Simulation service unavailable',
      details: error.message
    });
  }
});

// Simple test route
app.get('/', (req, res) => {
  res.send('NeuroAdaptive Learning API is running');
});

// Start server
const server = app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
  console.error('UNHANDLED REJECTION! 💥 Shutting down...');
  console.error(err.name, err.message);
  server.close(() => {
    process.exit(1);
  });
});
