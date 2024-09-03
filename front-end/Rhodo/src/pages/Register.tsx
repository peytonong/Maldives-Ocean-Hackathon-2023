// Register.tsx

import React from 'react';
import { Link, Route, RouteComponentProps } from 'react-router-dom';
import {
  IonButton,
  IonCheckbox,
  IonContent,
  IonHeader,
  IonInput,
  IonItem,
  IonLabel,
  IonPage,
  IonRouterOutlet,
  IonTitle,
} from '@ionic/react';
import './TestHome.css'; 

const Register: React.FC<RouteComponentProps> = () => {
  return (
    <IonPage>
      <IonHeader>
        <IonTitle>Register</IonTitle>
      </IonHeader>
      <IonContent>
        <div className="login-container">
          {/* Left Section: Logo and Form */}
          <div className="left-side">
            {/* Logo Goes Here */}
            <img src="/path/to/your/logo.png" alt="Logo" className="logo" />

            {/* Registration Form */}
            <form className="ion-padding">
              <IonItem>
                <IonLabel position="floating">Username</IonLabel>
                <IonInput />
              </IonItem>
              <IonItem>
                <IonLabel position="floating">Email</IonLabel>
                <IonInput type="email" />
              </IonItem>
              <IonItem>
                <IonLabel position="floating">Password</IonLabel>
                <IonInput type="password" />
              </IonItem>
              <IonButton className="ion-margin-top" type="submit" expand="block">
                Register
              </IonButton>
            </form>

            {/* Navigation link to Login page */}
            <p className="login-link">
              Already have an account?{' '}
              <Link to="/Login" className="login-link-text">
                Login here
              </Link>
            </p>
          </div>

          {/* Right Section: Placeholder */}
          <div className="right-side">
            {/* You can add an image or any other content here */}
          </div>
        </div>
      </IonContent>
    </IonPage>
  );
};

export default Register;
