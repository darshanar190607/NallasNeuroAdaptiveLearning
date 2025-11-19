import React, { useState, useEffect } from 'react';
import LoadingScreen from './components/ui/LoadingScreen.tsx';
import Header from './components/layout/Header.tsx';
import HomePage from './pages/home/HomePage.tsx';
import EnhancedCollaborationPage from './pages/collaboration/EnhancedCollaborationPage.tsx';
import RoadmapCreator from './pages/roadmap/RoadmapCreatorPage.tsx';
import PersonalDashboard from './pages/dashboard/PersonalDashboardPage.tsx';
import LoginPage from './pages/auth/LoginPage.tsx';
import SignupPage from './pages/auth/SignupPage.tsx';
import Footer from './components/layout/Footer.tsx';
import { BCIProvider } from './contexts/BCIContext.tsx';
import { AuthProvider, useAuth } from './contexts/AuthContext.tsx';
import AuthModal from './components/auth/AuthModal.tsx';
import OnboardingSurvey from './components/survey/OnboardingSurvey.tsx';

const AppContent: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(() => {
    const path = window.location.pathname;
    if (path === '/login') return 'login';
    if (path === '/signup') return 'signup';
    if (path === '/collaboration') return 'collaboration';
    if (path === '/roadmap') return 'roadmap';
    if (path === '/dashboard') return 'dashboard';
    return 'home';
  });
  const [showAuthModal, setShowAuthModal] = useState(false);
  const { user, isLoading: authLoading } = useAuth();

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const handlePopState = () => {
      const path = window.location.pathname;
      if (path === '/login') setCurrentPage('login');
      else if (path === '/signup') setCurrentPage('signup');
      else if (path === '/collaboration') setCurrentPage('collaboration');
      else if (path === '/roadmap') setCurrentPage('roadmap');
      else if (path === '/dashboard') setCurrentPage('dashboard');
      else setCurrentPage('home');
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const navigate = (page: string) => {
    setCurrentPage(page);
    const path = page === 'home' ? '/' : `/${page}`;
    window.history.pushState({}, '', path);
  };

  if (loading || authLoading) {
    return <LoadingScreen />;
  }

  // Show survey if user is new (just registered)
  if (user && user.needsSurvey) {
    return (
      <OnboardingSurvey onComplete={() => {
        updateUser({ needsSurvey: false });
        window.location.reload();
      }} />
    );
  }

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage />;
      case 'login':
        return <LoginPage />;
      case 'signup':
        return <SignupPage />;
      case 'collaboration':
        return <EnhancedCollaborationPage />;
      case 'roadmap':
        return <RoadmapCreator />;
      case 'dashboard':
        return <PersonalDashboard />;
      default:
        return <HomePage />;
    }
  };

  // Don't show header/footer for auth pages
  const isAuthPage = currentPage === 'login' || currentPage === 'signup';
  
  if (isAuthPage) {
    return (
      <div className="bg-gray-900 text-gray-100 min-h-screen font-sans">
        {renderPage()}
      </div>
    );
  }

  return (
    <BCIProvider>
      <div className="bg-gray-900 text-gray-100 min-h-screen font-sans flex flex-col">
        <Header 
          currentPage={currentPage} 
          setCurrentPage={navigate}
          onAuthClick={() => setShowAuthModal(true)}
        />
        <main className="flex-grow">
          {renderPage()}
        </main>
        <Footer />
        <AuthModal 
          isOpen={showAuthModal} 
          onClose={() => setShowAuthModal(false)} 
        />
      </div>
    </BCIProvider>
  );
};

const App: React.FC = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

export default App;