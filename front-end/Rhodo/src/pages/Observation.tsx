import React, { useEffect, useState } from 'react';
import {
  IonButton,
  IonContent,
  IonHeader,
  IonInput,
  IonItem,
  IonLabel,
  IonPage,
  IonTitle,
  IonImg,
} from '@ionic/react';
import { Position } from '@capacitor/geolocation';
import TopBar from '../components/topbar';

const Observation: React.FC = () => {
  const [formData, setFormData] = useState({
    locationDetails: '',
    date: '',
    lookDescription: '',
    waterTemperature: '',
    nutrientLevel: '',
    picture: null as File | null,
  });

  const [map, setMap] = useState<google.maps.Map | null>(null);
  const [userMarker, setUserMarker] = useState<google.maps.Marker | null>(null);

  useEffect(() => {
    const options = { timeout: 10000, enableHighAccuracy: true };

    const setupMap = async () => {
      try {
        const position = await getCurrentPosition(options);

        const locationDetails = `Latitude: ${position.coords.latitude}, Longitude: ${position.coords.longitude}`;
        setFormData((prevData) => ({
          ...prevData,
          locationDetails,
        }));

        const mapOptions: google.maps.MapOptions = {
          center: { lat: position.coords.latitude, lng: position.coords.longitude },
          zoom: 15,
        };

        const mapInstance = new google.maps.Map(
          document.getElementById('map') as HTMLElement,
          mapOptions
        );

        setMap(mapInstance);

        // Add a marker for the user's current position
        const marker = new google.maps.Marker({
          position: { lat: position.coords.latitude, lng: position.coords.longitude },
          map: mapInstance,
          title: 'Your Location',
        });

        setUserMarker(marker);
      } catch (error) {
        console.error('Could not get location', error);
      }
    };

    setupMap();
  }, []);

  const getCurrentPosition = (options: PositionOptions): Promise<Position> => {
    return new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(resolve, reject, options);
    });
  };

  const handleInputChange = (fieldName: string, value: string) => {
    setFormData((prevData) => ({
      ...prevData,
      [fieldName]: value,
    }));
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files && event.target.files[0];
    if (file) {
      setFormData((prevData) => ({
        ...prevData,
        picture: file,
      }));
    }
  };

  const handleSubmit = () => {
    console.log('Form Data:', formData);
    // Additional form submission logic
  };

  return (
    <IonPage>
      <TopBar></TopBar>
      <IonContent className=' p-5'>
        <div className='px-5'>
          <div className='p-3 h1 mt-5'>Observation Deck</div>
          <div className='px-3'>
            Rhodo uses AI Vision to recognize patterns and abnormalities of the marine and beaches from the observation submitted. Observation will then be used to further enhance, monitor and detect the prediction and dataset of the Maldives marine ecosystem.
            
          </div>
          <div className="my-5"></div>
          <hr/>
          <div className="observation-container">
            <div className='px-5 mx-5'>
            <div className='font-weight-bold'>Select location of the observation</div>
            <br/><br/>
            <div className='px-5'>
              <div id="map" style={{ height: '300px' }} data-tap-disabled="true"></div>

            </div>
            <br/><br/>
              <IonItem>
                <IonLabel position="floating">Date</IonLabel>
                <IonInput
                  type="date"
                  value={formData.date}
                  onIonChange={(e) => handleInputChange('date', e.detail.value!)}
                />
              </IonItem>

              <IonItem>
                <IonLabel position="floating">Description of the observation</IonLabel>
                <IonInput
                  value={formData.lookDescription}
                  onIonChange={(e) => handleInputChange('lookDescription', e.detail.value!)}
                />
              </IonItem>

              <IonItem>
                <IonLabel position="floating">Water temperature</IonLabel>
                <IonInput
                  type="number"
                  value={formData.waterTemperature}
                  onIonChange={(e) => handleInputChange('waterTemperature', e.detail.value!)}
                />
              </IonItem>

              <IonItem>
                <IonLabel position="floating">Nutrient level</IonLabel>
                <IonInput
                  value={formData.nutrientLevel}
                  onIonChange={(e) => handleInputChange('nutrientLevel', e.detail.value!)}
                />
              </IonItem>

              {/* File Input for Picture */}
              <IonItem>
                <IonLabel position="floating">Upload Picture</IonLabel>
                <br/><br/>
                <input
                  className="pictureInput"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                />
              </IonItem>
              <br/>
              {/* Display Uploaded Image */}
              <div className='d-flex justify-content-center'>
                {formData.picture && (
                  <IonItem>
                    <IonLabel position="stacked">Uploaded Image</IonLabel>
                    <IonImg src={URL.createObjectURL(formData.picture)} />
                  </IonItem>
                )}
              </div>
              

              <div className='d-flex justify-content-center'>
                <IonButton
                  className="ion-margin-top "
                  onClick={handleSubmit}
                >
                  Submit
                </IonButton>
              </div>
            </div>
            
            
          </div>
        </div>
        
      </IonContent>
    </IonPage>
  );
};

export default Observation;
