
import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-900 border-t border-gray-800">
      <div className="container mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div>
            <h3 className="text-xl font-bold text-white">NeuroBright</h3>
            <p className="mt-2 text-gray-400">Learn Smarter. Not Harder.</p>
          </div>
          <div>
            <h4 className="font-semibold text-gray-200">Platform</h4>
            <ul className="mt-4 space-y-2">
              <li><a href="#" className="text-gray-400 hover:text-white transition">Home</a></li>
              <li><a href="#" className="text-gray-400 hover:text-white transition">Courses</a></li>
              <li><a href="#" className="text-gray-400 hover:text-white transition">Roadmaps</a></li>
              <li><a href="#" className="text-gray-400 hover:text-white transition">Dashboard</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-gray-200">Company</h4>
            <ul className="mt-4 space-y-2">
              <li><a href="#" className="text-gray-400 hover:text-white transition">About Us</a></li>
              <li><a href="#" className="text-gray-400 hover:text-white transition">Careers</a></li>
              <li><a href="#" className="text-gray-400 hover:text-white transition">Press</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-gray-200">Connect</h4>
            <div className="flex mt-4 space-x-4">
              {/* SVG Icons for social media */}
            </div>
          </div>
        </div>
        <div className="mt-12 border-t border-gray-800 pt-8 text-center text-gray-500">
          <p>&copy; {new Date().getFullYear()} NeuroBright. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;