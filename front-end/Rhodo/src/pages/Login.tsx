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
import rightImage from '../assets/image/background.jpg';

const Login: React.FC<RouteComponentProps> = () => {
  return (
    <IonPage>
    <IonContent>
      <div className="login-container">
        {/* Left Section: Logo and Form */}
        <div className="left-side">
          {/* Logo Goes Here */}
          <img src="/path/to/your/logo.png" alt="Logo" className="logo" />

          {/* Login Form */}
          <form className="ion-padding">
            <IonItem>
              <IonLabel position="floating">Username</IonLabel>
              <IonInput />
            </IonItem>
            <IonItem>
              <IonLabel position="floating">Password</IonLabel>
              <IonInput type="password" />
            </IonItem>
            <IonItem lines="none">
              <IonLabel>Remember me</IonLabel>
              <IonCheckbox defaultChecked={true} slot="start" />
            </IonItem>
            <IonButton className="ion-margin-top" type="submit" expand="block">
              Login
            </IonButton>
          </form>
          <p className="register-link">
            Don't have an account?{' '}
              <Link to="/register" className="register-link-text">
                Register here
              </Link>
          </p>
        </div>

        {/* Right Section: Image */}
        <div className="right-side">
          {/*Image Goes Here */}
          <img src={rightImage} alt="Image" className="image" />
        </div>
      </div>
    </IonContent>
  </IonPage>
  );
};

export default Login;
