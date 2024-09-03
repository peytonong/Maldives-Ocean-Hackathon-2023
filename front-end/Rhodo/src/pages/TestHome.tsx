// TestHome.tsx
import React from 'react';
import { IonContent, IonHeader, IonPage, IonTitle, IonToolbar, IonList, IonItem, IonLabel, IonIcon, IonButton } from '@ionic/react';
import { Link } from 'react-router-dom';
import './TestHome.css'; // Import the CSS file
import rightImage from '../assets/image/background.jpg';
import { camera, handRightOutline, home, informationCircle, people } from 'ionicons/icons';

const TestHome: React.FC = () => {
  return (
    <IonPage>
    <IonContent>
      <div className="login-container">

        {/* Right Section: Image */}
        <div className="right-section">
        <div className='rightWord'>
            <h2>Welcome!</h2>
            <p>
              Thank you for visiting our platform. Explore and engage with our community.
            </p>
            <IonButton expand="full" size="small">
              See More
            </IonButton>
          </div>

          {/* Observation Section */}
          <div className='rightWord'>
            <h2>Observation</h2>
            <p>
              Explore the fascinating world of observations and share your experiences with others.
            </p>
            <IonButton expand="full" size="small">
              See More
            </IonButton>
          </div>

          {/* Community Section */}
          <div className='rightWord'>
            <h2>Community</h2>
            <p>
              Connect with like-minded individuals, share your insights, and be part of our community.
            </p>
            <IonButton expand="full" size="small">
              See More
            </IonButton>
          </div>

          {/* Collaborate Section */}
          <div className='rightWord'>
            <h2>Collaborate</h2>
            <p>
              Collaborate with others, exchange ideas, and contribute to meaningful projects.
            </p>
            <IonButton expand="full" size="small">
              See More
            </IonButton>
          </div>
        </div>
        </div>
    </IonContent>
  </IonPage>
  );
};

export default TestHome;
