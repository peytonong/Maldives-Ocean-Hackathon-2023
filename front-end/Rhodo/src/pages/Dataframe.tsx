// DataFrame.tsx
import React, { useState } from 'react';
import {
  IonTextarea,
  IonDatetime,
  IonInput,
  IonButton,
  IonAlert,
  IonPage,
  IonContent
} from '@ionic/react';
import './DataFrame.css'; // Import the Ionic CSS
import TopBar from '../components/topbar';

const DataFrame: React.FC = () => {
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleSubmit = () => {
    // Handle form submission logic
    // For simplicity, let's just set a flag to show the confirmation message
    setIsSubmitted(true);
  };

  return (
    <IonPage>
        <TopBar></TopBar>
        <IonContent className='p-5'>
          <div className='px-5'>
          <div className='p-3 h1 mt-5'>Data Form</div>
              <div className='px-3'>
              Contributes towards the Maldivian Marine Ecosystem by contributing any data source. These data source will be validated and analyzed before being released to the public and to enhance the current Red Algae Monitoring and Detection system.
              
            </div>
            <div className="data-frame">
            <div className="my-5"></div>
           <hr/>
            <div className="data-frame-container mt-5">
                <div className="form-group">
                <label>Description of the Data</label>
                <IonTextarea placeholder="Enter a detailed description" className="ion-textarea" />
                </div>
                <div className="form-group">
                <label>Data Uploads</label>
                <input type="file" className="form-control" />
                </div>
                <div className="form-group">
                <label>Data Source</label>
                <IonInput placeholder="Enter where your data is sourced from" className="ion-input" />
                </div>
                <div className="form-group">
                <label>Contact Information</label>
                <IonInput placeholder="Enter contact information" className="ion-input" />
                </div>
                <IonButton expand="full" onClick={handleSubmit}>Submit</IonButton>

                {/* Ionic Alert for the confirmation message */}
                <IonAlert
                isOpen={isSubmitted}
                onDidDismiss={() => setIsSubmitted(false)}
                header={'Thank You!'}
                message={'Your data has been successfully submitted.'}
                buttons={['OK']}
                />
            </div>
            </div>
          </div>
            
        </IonContent>
    </IonPage>
  );
};

export default DataFrame;
