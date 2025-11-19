import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import collaborationService from '../../services/collaborationService';
import { UsersIcon, HoloRoomIcon, MissionIcon, CopyIcon, TrophyIcon } from '../../components/ui/Icons';

type Participant = {
  _id: string;
  user: {
    _id: string;
    name: string;
    email: string;
  };
  points: number;
  studyTime: number;
  joinedAt: string;
};

type ChatMessage = {
  sender: string;
  senderName: string;
  message: string;
  timestamp: string;
};

type Question = {
  question: string;
  answers: string[];
  correct: number;
};

type View = 'intro' | 'lobby' | 'session';

const EnhancedCollaborationPage: React.FC = () => {
  const { user } = useAuth();
  const [view, setView] = useState<View>('intro');
  const [topic, setTopic] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [sessionLink, setSessionLink] = useState('');
  const [linkCopied, setLinkCopied] = useState(false);
  
  const [participants, setParticipants] = useState<Participant[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [quiz, setQuiz] = useState<Question[] | null>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState<Record<number, number>>({});
  const [showCorrectAnswer, setShowCorrectAnswer] = useState(false);
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!user) {
      setView('intro');
    }
  }, [user]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleStartSession = async () => {
    if (!topic.trim() || !user) return;
    setIsLoading(true);
    setError('');
    
    try {
      // Generate quiz
      const quizData = await collaborationService.generateQuiz(topic);
      
      // Create session
      const sessionData = await collaborationService.createSession(topic, quizData.quiz);
      
      setQuiz(quizData.quiz);
      setSessionId(sessionData.session.sessionId);
      setSessionLink(`${window.location.origin}/collaboration/${sessionData.session.sessionId}`);
      setParticipants(sessionData.session.participants);
      setCurrentQuestionIndex(0);
      setUserAnswers({});
      setChatMessages([]);
      setView('session');
    } catch (error: any) {
      setError(error.message || 'Failed to create session');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnswerQuestion = async (questionIndex: number, answerIndex: number) => {
    if (userAnswers[questionIndex] !== undefined || !sessionId) return;

    try {
      const result = await collaborationService.submitAnswer(sessionId, questionIndex, answerIndex);
      setUserAnswers(prev => ({ ...prev, [questionIndex]: answerIndex }));
      setParticipants(result.session.participants);

      // Show correct answer after delay
      setTimeout(() => setShowCorrectAnswer(true), 3000);
      setTimeout(() => {
        setShowCorrectAnswer(false);
        if (currentQuestionIndex < quiz!.length - 1) {
          setCurrentQuestionIndex(prev => prev + 1);
        }
      }, 5000);
    } catch (error: any) {
      console.error('Answer submission error:', error);
    }
  };

  const handleSendMessage = async (text: string) => {
    if (!text.trim() || !sessionId) return;
    
    try {
      const result = await collaborationService.sendMessage(sessionId, text);
      setChatMessages(prev => [...prev, {
        sender: result.chatMessage.sender,
        senderName: result.chatMessage.senderName,
        message: result.chatMessage.message,
        timestamp: result.chatMessage.timestamp
      }]);
    } catch (error: any) {
      console.error('Send message error:', error);
    }
  };

  const handleCopyLink = () => {
    navigator.clipboard.writeText(sessionLink);
    setLinkCopied(true);
    setTimeout(() => setLinkCopied(false), 2000);
  };

  const renderIntro = () => (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      <div className="container mx-auto px-6 py-12">
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-white mb-6">Collaboration Nexus</h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Join forces with learners worldwide. Create study sessions, solve quizzes together, and accelerate your learning through collaboration.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
          <div className="bg-gray-800/50 backdrop-blur-lg border border-gray-700 rounded-2xl p-8 text-center">
            <HoloRoomIcon className="w-16 h-16 text-cyan-400 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-white mb-4">Virtual Study Rooms</h3>
            <p className="text-gray-400">Create or join study sessions on any topic with real-time collaboration features.</p>
          </div>
          
          <div className="bg-gray-800/50 backdrop-blur-lg border border-gray-700 rounded-2xl p-8 text-center">
            <MissionIcon className="w-16 h-16 text-purple-400 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-white mb-4">Interactive Quizzes</h3>
            <p className="text-gray-400">AI-generated quizzes that adapt to your chosen topics and learning objectives.</p>
          </div>
          
          <div className="bg-gray-800/50 backdrop-blur-lg border border-gray-700 rounded-2xl p-8 text-center">
            <TrophyIcon className="w-16 h-16 text-yellow-400 mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-white mb-4">Gamified Learning</h3>
            <p className="text-gray-400">Earn points, compete with peers, and track your progress in real-time leaderboards.</p>
          </div>
        </div>

        <div className="text-center">
          {user ? (
            <button
              onClick={() => setView('lobby')}
              className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-500 text-white font-bold rounded-full text-lg transform hover:scale-105 transition-transform duration-300 shadow-lg"
            >
              Start Collaborating
            </button>
          ) : (
            <div className="space-y-4">
              <p className="text-gray-400 mb-4">Sign in to start collaborating with other learners</p>
              <div className="space-x-4">
                <a
                  href="/login"
                  className="inline-block px-6 py-3 bg-gradient-to-r from-orange-500 to-red-500 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-red-600 transition-all"
                >
                  Sign In
                </a>
                <a
                  href="/signup"
                  className="inline-block px-6 py-3 border border-gray-600 text-gray-300 font-semibold rounded-lg hover:bg-gray-800 transition-all"
                >
                  Sign Up
                </a>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderLobby = () => (
    <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-lg z-50 flex items-center justify-center p-4">
      <div className="w-full max-w-lg bg-gray-800/90 backdrop-blur-lg border border-gray-700 rounded-2xl shadow-2xl p-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-white mb-4">Create Study Session</h2>
          <p className="text-gray-400">What topic would you like to explore today?</p>
        </div>

        {error && (
          <div className="bg-red-500/20 border border-red-500 text-red-300 px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        <form onSubmit={(e) => { e.preventDefault(); handleStartSession(); }} className="space-y-6">
          <div>
            <label htmlFor="topic" className="block text-sm font-medium text-gray-300 mb-2">
              Study Topic
            </label>
            <input
              type="text"
              id="topic"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g., Quantum Physics, Machine Learning, History..."
              className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              required
            />
          </div>

          <div className="flex space-x-4">
            <button
              type="button"
              onClick={() => setView('intro')}
              className="flex-1 px-6 py-3 border border-gray-600 text-gray-300 font-semibold rounded-lg hover:bg-gray-700 transition-all"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading || !topic.trim()}
              className="flex-1 bg-gradient-to-r from-blue-500 to-purple-500 text-white font-bold py-3 px-6 rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Creating...
                </div>
              ) : (
                'Create Session'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );

  const renderSession = () => (
    <div className="min-h-screen bg-gray-900 text-gray-200 p-4">
      <header className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold text-white">Study Session</h1>
          <p className="text-gray-400">Topic: <span className="font-semibold text-blue-400">{topic}</span></p>
        </div>
        <button 
          onClick={() => setView('intro')} 
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white font-semibold rounded-lg transition-all"
        >
          Exit Session
        </button>
      </header>
      
      {sessionLink && (
        <div className="flex items-center gap-4 mb-6 bg-gray-800/50 p-4 rounded-lg border border-gray-700">
          <span className="text-gray-400 font-semibold">Invite Link:</span>
          <input 
            type="text" 
            readOnly 
            value={sessionLink} 
            className="flex-grow bg-gray-700 text-gray-300 rounded px-3 py-2 border border-gray-600"
          />
          <button 
            onClick={handleCopyLink} 
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-2 rounded-lg transition-all"
          >
            <CopyIcon className="w-4 h-4" />
            {linkCopied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Participants Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-gray-800/50 backdrop-blur-lg border border-gray-700 rounded-lg p-4">
            <h2 className="text-lg font-bold mb-4 text-white flex items-center">
              <UsersIcon className="w-5 h-5 mr-2" />
              Participants ({participants.length})
            </h2>
            <div className="space-y-3">
              {participants.map(p => (
                <div key={p._id} className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold">
                    {p.user.name.charAt(0)}
                  </div>
                  <div>
                    <p className="font-semibold text-gray-300">{p.user.name}</p>
                    <p className="text-xs text-gray-500">{p.points} points</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        
        {/* Quiz Area */}
        <div className="lg:col-span-2 bg-gray-800/50 backdrop-blur-lg border border-gray-700 rounded-lg p-6">
          {quiz && (
            <QuizArea
              quiz={quiz}
              currentQuestionIndex={currentQuestionIndex}
              onAnswer={handleAnswerQuestion}
              userAnswers={userAnswers}
              showCorrectAnswer={showCorrectAnswer}
            />
          )}
        </div>
        
        {/* Chat Panel */}
        <div className="lg:col-span-1 bg-gray-800/50 backdrop-blur-lg border border-gray-700 rounded-lg p-4 flex flex-col h-96">
          <ChatPanel 
            messages={chatMessages} 
            onSendMessage={handleSendMessage} 
            chatEndRef={chatEndRef} 
          />
        </div>
      </div>
    </div>
  );

  if (view === 'lobby') return renderLobby();
  if (view === 'session') return renderSession();
  return renderIntro();
};

// Quiz Component
const QuizArea: React.FC<{
  quiz: Question[];
  currentQuestionIndex: number;
  onAnswer: (q: number, a: number) => void;
  userAnswers: Record<number, number>;
  showCorrectAnswer: boolean;
}> = ({ quiz, currentQuestionIndex, onAnswer, userAnswers, showCorrectAnswer }) => {
  const question = quiz[currentQuestionIndex];
  if (!question) return <div className="text-center p-8"><h2 className="text-2xl font-bold">Quiz Complete!</h2></div>;

  const userAnswer = userAnswers[currentQuestionIndex];

  return (
    <div className="flex flex-col h-full">
      <div className="mb-6">
        <p className="text-sm text-gray-400 mb-2">Question {currentQuestionIndex + 1} of {quiz.length}</p>
        <h3 className="text-2xl font-bold text-white">{question.question}</h3>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-grow">
        {question.answers.map((answer, index) => {
          const isSelected = userAnswer === index;
          const isCorrect = question.correct === index;
          let buttonClass = 'bg-gray-700 hover:bg-gray-600 border-gray-600';
          
          if (showCorrectAnswer) {
            if (isCorrect) buttonClass = 'bg-green-600 border-green-500';
            else if (isSelected) buttonClass = 'bg-red-600 border-red-500';
            else buttonClass = 'bg-gray-800 text-gray-500 border-gray-700';
          } else if (isSelected) {
            buttonClass = 'bg-blue-600 border-blue-500 ring-2 ring-blue-400';
          }

          return (
            <button
              key={index}
              onClick={() => onAnswer(currentQuestionIndex, index)}
              disabled={userAnswer !== undefined}
              className={`p-4 rounded-lg text-left transition-all border ${buttonClass} disabled:cursor-not-allowed`}
            >
              <span className="font-semibold">{answer}</span>
            </button>
          );
        })}
      </div>
      
      <div className="mt-6 text-center text-gray-500 text-sm">
        {userAnswer === undefined && "Select an answer to continue"}
        {userAnswer !== undefined && !showCorrectAnswer && "Waiting for results..."}
        {showCorrectAnswer && "Moving to next question..."}
      </div>
    </div>
  );
};

// Chat Component
const ChatPanel: React.FC<{
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  chatEndRef: React.RefObject<HTMLDivElement>;
}> = ({ messages, onSendMessage, chatEndRef }) => {
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const input = e.currentTarget.message as HTMLInputElement;
    onSendMessage(input.value);
    input.value = '';
  };

  return (
    <>
      <h2 className="text-lg font-bold mb-3 text-white">Chat</h2>
      <div className="flex-grow overflow-y-auto mb-3 space-y-3">
        {messages.map((msg, index) => (
          <div key={index} className="text-sm">
            <p className="font-semibold text-blue-400">{msg.senderName}</p>
            <p className="text-gray-300">{msg.message}</p>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input 
          name="message" 
          type="text" 
          placeholder="Type a message..." 
          className="flex-grow bg-gray-700 text-white rounded px-3 py-2 border border-gray-600 focus:outline-none focus:ring-1 focus:ring-blue-500" 
        />
        <button 
          type="submit" 
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded transition-all"
        >
          Send
        </button>
      </form>
    </>
  );
};

export default EnhancedCollaborationPage;