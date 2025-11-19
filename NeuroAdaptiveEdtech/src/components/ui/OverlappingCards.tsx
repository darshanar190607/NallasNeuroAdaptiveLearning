

import React, { useRef, useState, useEffect } from 'react';

const cardData = [
  {
    title: 'Adaptive Learning Paths',
    description: 'Our AI analyzes your survey results and learning patterns to generate a curriculum perfectly tailored to your strengths and weaknesses.',
    color: 'from-blue-500 to-cyan-400',
    img: 'https://picsum.photos/seed/path/500/300'
  },
  {
    title: 'BCI Engagement Monitoring',
    description: 'Using your webcam, our system simulates BCI technology to detect signs of drowsiness or distraction, dynamically adjusting content to keep you engaged.',
    color: 'from-purple-500 to-indigo-400',
    img: 'https://picsum.photos/seed/bci/500/300'
  },
  {
    title: 'Interactive 3D & AR Content',
    description: 'When you lose focus, we introduce stunning 3D visualizations, AR experiences, and gamified quizzes to reignite your curiosity and make learning fun.',
    color: 'from-pink-500 to-rose-400',
    img: 'https://picsum.photos/seed/arvr/500/300'
  },
  {
    title: 'Collaborative Leaderboards',
    description: 'Connect with friends, tackle challenges together, and climb the leaderboards. Earn points and rewards as you master new skills.',
    color: 'from-emerald-500 to-lime-400',
    img: 'https://picsum.photos/seed/collab/500/300'
  },
];

const OverlappingCards: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollProgress, setScrollProgress] = useState(0);

  const handleScroll = () => {
    const container = containerRef.current;
    if (container) {
      const { top, height } = container.getBoundingClientRect();
      const viewportHeight = window.innerHeight;
      
      // Calculate how much of the container is visible and has been scrolled past
      const distance = viewportHeight - top;
      const totalScrollableHeight = height + viewportHeight;

      let progress = 0;
      if (top < viewportHeight && top > -height) {
          progress = Math.max(0, Math.min(1, distance / totalScrollableHeight * 1.5)); // Multiply by 1.5 to speed up animation
      }
      setScrollProgress(progress);
    }
  };

  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <section className="relative bg-gray-900 py-24">
       <div className="container mx-auto px-6 text-center">
            <h2 className="text-4xl font-extrabold text-white">How NeuroBright Works</h2>
            <p className="mt-4 text-lg text-gray-400 max-w-2xl mx-auto">
                A revolutionary four-step process to redefine your learning experience.
            </p>
        </div>
      <div ref={containerRef} className="relative h-[300vh] mt-16">
        <div className="sticky top-1/4">
          {cardData.map((card, index) => {
            const progressPerCard = 1 / cardData.length;
            const cardStartProgress = index * progressPerCard;
            const cardProgress = Math.max(0, Math.min(1, (scrollProgress - cardStartProgress) / progressPerCard));

            const scale = 1 - (cardData.length - 1 - index) * 0.05 + cardProgress * 0.05;
            const translateY = -index * 2 + cardProgress * 2;
            const opacity = cardProgress > 0.5 ? 2 - cardProgress * 2 : 1;

            return (
              <div
                key={index}
                className="absolute w-full h-full flex justify-center items-start"
                style={{
                  transform: `scale(${scale}) translateY(${translateY}rem)`,
                  zIndex: index,
                  opacity: index === 0 ? 1 : Math.max(0, cardProgress * 2 - 1),
                }}
              >
                <div className={`
                    w-full max-w-4xl rounded-2xl bg-gray-800 border border-gray-700
                    p-8 shadow-2xl shadow-black/50 transition-all duration-300 ease-out
                    flex flex-col md:flex-row gap-8 items-center
                `}>
                    <div className="md:w-1/2 text-left">
                        <h3 className={`text-3xl font-bold bg-gradient-to-r ${card.color} bg-clip-text text-transparent`}>{card.title}</h3>
                        <p className="mt-4 text-gray-300">{card.description}</p>
                    </div>
                     <div className="md:w-1/2">
                        <img src={card.img} alt={card.title} className="rounded-lg shadow-lg" />
                    </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default OverlappingCards;