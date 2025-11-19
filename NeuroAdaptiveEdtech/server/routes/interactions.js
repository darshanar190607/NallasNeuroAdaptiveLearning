import express from 'express';
import UserInteraction from '../models/UserInteraction.js';

const router = express.Router();

// Log a new user interaction
router.post('/', async (req, res) => {
  try {
    const { userId, action, details } = req.body;
    
    const interaction = new UserInteraction({
      userId,
      action,
      details
    });

    await interaction.save();
    res.status(201).json({ message: 'Interaction logged successfully', interaction });
  } catch (error) {
    console.error('Error logging interaction:', error);
    res.status(500).json({ message: 'Error logging interaction' });
  }
});

// Get all interactions for a user
router.get('/user/:userId', async (req, res) => {
  try {
    const interactions = await UserInteraction.find({ userId: req.params.userId })
      .sort({ timestamp: -1 });
    res.json(interactions);
  } catch (error) {
    console.error('Error fetching interactions:', error);
    res.status(500).json({ message: 'Error fetching interactions' });
  }
});

export default router;
