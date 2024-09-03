// MyComponent.tsx

import React from 'react';
import './Component.css';

// interface ComponentProps {
//   name: string;
//   imageUrl: string;
//   description: string;
// }

const Component = () => {
    return (
      <div className="container">
        {/* Sidebar */}
        <div className="sidebar">
          <img src="" alt="Profile" />
        </div>
  
        {/* Main Content */}
        <div className="main-content">
          <h1>Name</h1>
          <p>Description goes here...</p>
        </div>
      </div>
    );
  };

export default Component;
