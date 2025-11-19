import React from 'react';
import { WelcomeButton, SearchBar } from '../../components/ui/Icons.tsx';
import CourseChat from '../../features/course-chat/CourseChat.tsx';
import OverlappingCards from '../../components/ui/OverlappingCards.tsx';
import AdaptiveLearning from '../../components/learning/AdaptiveLearning.tsx';
import { useAuth } from '../../contexts/AuthContext.tsx';


const HomePage: React.FC = () => {
  const { user } = useAuth();

  return (
    <div>
      {/* Hero Section */}
      <section className="relative container mx-auto px-6 py-20 md:py-32 flex flex-col md:flex-row items-center overflow-hidden">
        <div className="absolute top-0 left-0 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-br from-red-500/20 to-orange-500/10 rounded-full blur-3xl -z-0"></div>
        <div className="absolute bottom-0 right-0 translate-x-1/2 translate-y-1/2 w-[500px] h-[500px] bg-gradient-to-tl from-purple-600/20 to-blue-500/10 rounded-full blur-3xl -z-0"></div>
        
        <div className="md:w-1/2 lg:w-3/5 text-center md:text-left z-10">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-white leading-tight tracking-tighter">
            Learn Smarter with
            <span className="block bg-gradient-to-r from-red-500 to-orange-500 bg-clip-text text-transparent mt-2">AI & BCI Technology</span>
          </h1>
          <p className="mt-6 text-lg text-gray-400 max-w-xl mx-auto md:mx-0">
            Unlock your full potential with a learning platform that adapts to you. NeuroBright uses advanced AI and brain-computer interface simulations to create a truly personalized and engaging educational journey.
          </p>
          <div className="mt-10 flex justify-center md:justify-start">
            <WelcomeButton />
          </div>
        </div>
        <div className="md:w-1/2 lg:w-2/5 mt-12 md:mt-0 flex justify-center items-center z-10">
            <div 
              className="relative w-80 h-80 lg:w-96 lg:h-96 rounded-full border-4 border-gray-700 shadow-2xl shadow-orange-500/20 transform hover:scale-105 transition-transform duration-500 overflow-hidden"
              role="img"
              aria-label="AI Brain"
            >
                <img src="https://tse3.mm.bing.net/th/id/OIP.CSA6xvkDXcdr8U7v3gnplQAAAA?cb=ucfimg2ucfimg=1&rs=1&pid=ImgDetMain&o=7&rm=3" alt="AI Brain" className="w-full h-full object-cover" />
            </div>
        </div>
      </section>

      {/* How It Works Section */}
      <OverlappingCards />

      {/* Sub Navbar */}
      <section className="sticky top-[77px] z-40 bg-gray-800/60 backdrop-blur-md border-y border-gray-700">
          <div className="container mx-auto px-6 py-3 flex justify-between items-center">
              <div className="flex space-x-6">
                <a href="#" className="text-white font-semibold border-b-2 border-orange-500 pb-1">Course Learning</a>
                <a href="#" className="text-gray-400 hover:text-white transition">Roadmap Creator</a>
                <a href="#" className="text-gray-400 hover:text-white transition">Collab</a>
                <a href="#" className="text-gray-400 hover:text-white transition">Dashboard</a>
              </div>
              <SearchBar />
          </div>
      </section>

      {/* Adaptive Learning Section */}
      {user && !user.needsSurvey ? (
        <section className="py-16 bg-gray-800/50">
          <div className="container mx-auto px-6">
            <h2 className="text-3xl font-bold text-white text-center mb-12">
              Your Personalized Learning Experience
            </h2>
            <AdaptiveLearning />
          </div>
        </section>
      ) : (
        <CourseChat />
      )}
    </div>
  );
};

export default HomePage;