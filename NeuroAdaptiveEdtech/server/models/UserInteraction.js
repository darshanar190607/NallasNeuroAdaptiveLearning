import mongoose from 'mongoose';

const userInteractionSchema = new mongoose.Schema({
  userId: {
    type: String,
    required: true
  },
  action: {
    type: String,
    required: true
  },
  details: {
    type: Object
  },
  timestamp: {
    type: Date,
    default: Date.now
  }
});

export default mongoose.model('UserInteraction', userInteractionSchema);
