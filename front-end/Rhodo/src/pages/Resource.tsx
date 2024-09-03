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
  IonIcon,
} from '@ionic/react';
import { Position } from '@capacitor/geolocation';
import TopBar from '../components/topbar';
import VisualizeStat from '../visualization/VisualizeStat';
import { arrowForwardOutline, eyeOutline, layersOutline, codeSlashOutline, codeSlash } from 'ionicons/icons';


const Resource: React.FC = () => {
  
  return (
    <IonPage>
      <TopBar></TopBar>
      <IonContent>
        <div className='px-5'>
          <div className='px-3 h1 mt-4'>Resources</div>
          <div className='px-3'>Get data, statistics and contribute to the open Maldives Marine data set</div>
          <hr/>
          <div className='d-flex justify-content-center' >
              <VisualizeStat width='800px' height='400px' imgsrc='https://public.tableau.com/static/images/TE/TEST_16971692142830/Dashboard1/1_rss.png' 
              staticimg='https://public.tableau.com/static/images/Fi/Final_16974958513780/Factors/1.png'
              statname='Final_16974958513780/Factors'
              path=''
              ></VisualizeStat>
            <div className='d-flex flex-column align-items-stretch bg-primary' >
            </div>
            <div className='d-flex flex-column justify-content-center align-items-center p-5'>
              <button className='h1 rounded-circle p-4 text-center bg-success text-white'
                onClick={() => {
                  window.location.href = 'resource/stathub';
                }}>
                <IonIcon icon={arrowForwardOutline}></IonIcon>
              </button>
              <div className='h5'>More Stats</div>

            </div>
          </div>
          <br/>
          <br/>
          <br/>
          <br/>
          <br/>
          <div className='d-flex p-5 shadow p-3 mb-5 bg-body rounded mx-5'>
              <div className='p-5'>
                <IonIcon icon={eyeOutline} className='h1'></IonIcon>
              </div>
              <div className='d-inline-flex flex-column '>
                <div className='h1'>Observation Deck</div>
                <div>Contribute towards creating a sustainable ecosystem. Upload your findings or observations at your location</div>
                <div className='py-2'></div>
                <button className='h4 shadow rounded-pill p-3 btn btn-success w-25'
                  onClick={() => {
                    window.location.href = 'resource/observation';
                  }}>Make Observation</button>
              </div>
          </div>
          <div className='d-flex p-5 shadow p-3 mb-5 bg-body rounded mx-5'>
              <div className='p-5'>
                <IonIcon icon={layersOutline} className='h1'></IonIcon>
              </div>
              <div className='d-inline-flex flex-column '>
                <div className='h1'>Data Form</div>
                <div>Contribute towards creating a sustainable ecosystem. Upload any marine related data and contribute towards the open Maldives data source</div>
                <div className='py-2'></div>
                <button className='h4 shadow rounded-pill p-3 btn btn-success w-25' 
                  onClick={() => {
                    window.location.href = 'resource/dataframe';
                  }}>Submit Data</button>
              </div>
          </div>
          <div className='d-flex p-5 shadow p-3 mb-5 bg-body rounded mx-5'>
              <div className='p-5'>
                <IonIcon icon={codeSlashOutline} className='h1'></IonIcon>
              </div>
              <div className='d-inline-flex flex-column '>
                <div className='h1'>Documentation</div>
                <div>Get access to the open Maldives Marine data source from the community to the community. Get via API access or via your preferred data format</div>
                <div className='py-2'></div>
                <button className='h4 shadow rounded-pill p-3 btn btn-success w-25' 
                  onClick={() => {
                    window.location.href = 'resource/documentation';
                  }}>View Documentation</button>
              </div>
          </div>
        </div>
        
      </IonContent>
      
    </IonPage>
  );
};

export default Resource;
