import React from 'react';
import { BrainCircuitIcon } from '../ui/Icons.tsx';
import { useAuth } from '../../contexts/AuthContext.tsx';

type NavLinkProps = {
    onClick: () => void;
    isActive: boolean;
    children: React.ReactNode;
};

const NavLink: React.FC<NavLinkProps> = ({ onClick, isActive, children }) => (
    <button onClick={onClick} className="relative text-gray-300 hover:text-white transition-colors duration-300 group">
        {children}
        <span className={`absolute -bottom-1 left-0 w-full h-0.5 bg-gradient-to-r from-red-500 to-orange-500 transition-transform duration-300 ease-out origin-center ${isActive ? 'scale-x-100' : 'scale-x-0 group-hover:scale-x-100'}`}></span>
    </button>
);

type HeaderProps = {
    currentPage: string;
    setCurrentPage: (page: string) => void;
    onAuthClick?: () => void;
};

const Header: React.FC<HeaderProps> = ({ currentPage, setCurrentPage, onAuthClick }) => {
  const { user, logout } = useAuth();

  return (
    <header className="sticky top-0 z-50 bg-[#0d1117] border-b border-gray-800">
      <div className="container mx-auto px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <BrainCircuitIcon className="w-8 h-8 text-orange-500"/>
          <a href="#" onClick={(e) => { e.preventDefault(); setCurrentPage('home'); }} className="text-2xl font-bold bg-gradient-to-r from-red-500 to-orange-500 bg-clip-text text-transparent">
            NeuroBright
          </a>
        </div>
        <nav className="hidden md:flex items-center space-x-8">
          <NavLink onClick={() => setCurrentPage('home')} isActive={currentPage === 'home'}>Home</NavLink>
          <NavLink onClick={() => setCurrentPage('roadmap')} isActive={currentPage === 'roadmap'}>Roadmap Creator</NavLink>
          <NavLink onClick={() => setCurrentPage('collaboration')} isActive={currentPage === 'collaboration'}>Collaboration</NavLink>
          {user && <NavLink onClick={() => setCurrentPage('dashboard')} isActive={currentPage === 'dashboard'}>Dashboard</NavLink>}
        </nav>
        <div className="flex items-center space-x-4">
          {user ? (
            <div className="flex items-center space-x-4">
              <span className="text-gray-300">Welcome, {user.name}</span>
              <button 
                onClick={logout}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Logout
              </button>
            </div>
          ) : (
            <div className="flex items-center space-x-3">
              <button 
                onClick={() => setCurrentPage('login')}
                className="px-4 py-2 text-gray-300 hover:text-white transition-colors"
              >
                Sign In
              </button>
              <button 
                onClick={() => setCurrentPage('signup')}
                className="px-4 py-2 bg-gradient-to-r from-orange-500 to-red-500 text-white rounded-lg hover:from-orange-600 hover:to-red-600 transition-all"
              >
                Sign Up
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;