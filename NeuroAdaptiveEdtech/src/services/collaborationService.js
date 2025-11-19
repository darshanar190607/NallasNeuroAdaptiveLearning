const API_BASE = 'http://localhost:5001/api';

class CollaborationService {
  constructor() {
    this.token = localStorage.getItem('token');
  }

  getAuthHeaders() {
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.token}`
    };
  }

  async createSession(topic, quiz) {
    try {
      const response = await fetch(`${API_BASE}/collaboration/sessions`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({ topic, quiz })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Failed to create session');
      }

      return data;
    } catch (error) {
      console.error('Create session error:', error);
      throw error;
    }
  }

  async joinSession(sessionId) {
    try {
      const response = await fetch(`${API_BASE}/collaboration/sessions/${sessionId}/join`, {
        method: 'POST',
        headers: this.getAuthHeaders()
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Failed to join session');
      }

      return data;
    } catch (error) {
      console.error('Join session error:', error);
      throw error;
    }
  }

  async getSession(sessionId) {
    try {
      const response = await fetch(`${API_BASE}/collaboration/sessions/${sessionId}`, {
        headers: this.getAuthHeaders()
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Failed to get session');
      }

      return data;
    } catch (error) {
      console.error('Get session error:', error);
      throw error;
    }
  }

  async submitAnswer(sessionId, questionIndex, answerIndex) {
    try {
      const response = await fetch(`${API_BASE}/collaboration/sessions/${sessionId}/answer`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({ questionIndex, answerIndex })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Failed to submit answer');
      }

      return data;
    } catch (error) {
      console.error('Submit answer error:', error);
      throw error;
    }
  }

  async sendMessage(sessionId, message) {
    try {
      const response = await fetch(`${API_BASE}/collaboration/sessions/${sessionId}/chat`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({ message })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Failed to send message');
      }

      return data;
    } catch (error) {
      console.error('Send message error:', error);
      throw error;
    }
  }

  async updateStudyTime(sessionId, studyTime) {
    try {
      const response = await fetch(`${API_BASE}/collaboration/sessions/${sessionId}/study-time`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({ studyTime })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || 'Failed to update study time');
      }

      return data;
    } catch (error) {
      console.error('Update study time error:', error);
      throw error;
    }
  }

  // Generate quiz using AI
  async generateQuiz(topic) {
    try {
      // This would typically call an AI service
      // For now, return a mock quiz
      const mockQuiz = [
        {
          question: `What is a key concept in ${topic}?`,
          answers: ["Option A", "Option B", "Option C", "Option D"],
          correct: 2
        },
        {
          question: `Which principle applies to ${topic}?`,
          answers: ["Principle 1", "Principle 2", "Principle 3", "Principle 4"],
          correct: 1
        },
        {
          question: `How does ${topic} relate to modern applications?`,
          answers: ["Through A", "Through B", "Through C", "Through D"],
          correct: 0
        },
        {
          question: `What is the future of ${topic}?`,
          answers: ["Future A", "Future B", "Future C", "Future D"],
          correct: 3
        },
        {
          question: `Which challenge exists in ${topic}?`,
          answers: ["Challenge A", "Challenge B", "Challenge C", "Challenge D"],
          correct: 1
        }
      ];

      return { quiz: mockQuiz };
    } catch (error) {
      console.error('Generate quiz error:', error);
      throw error;
    }
  }
}

export default new CollaborationService();