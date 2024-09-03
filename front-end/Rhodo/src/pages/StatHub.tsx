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
import TopBar from '../components/topbar';
import { downloadOutline, alertCircleOutline} from 'ionicons/icons';
import VisualizeStat from '../visualization/VisualizeStat';

const StatHub: React.FC = () => {
  
  return (
    <IonPage>
      <TopBar></TopBar>
      <IonContent>
        <div className='px-5'>
            <div className='px-3 h1 mt-4'>Stat Hub</div>
            <div className='px-3'>Open statistics and data for the community by the community</div>
            <hr/>
            <div className='d-flex'>
                <VisualizeStat width='1000px' height='450px' imgsrc='https://public.tableau.com/static/images/BX/BXHDNSGRJ/1_rss.png' 
                    staticimg='https://public.tableau.com/static/images/BX/BXHDNSGRJ/1.png'
                    statname='' path='shared/BXHDNSGRJ'></VisualizeStat>    

                <VisualizeStat width='1000px' height='450px' imgsrc='https://public.tableau.com/static/images/TE/TEST_16971692142830/Dashboard1/1_rss.png' 
                    staticimg='https://public.tableau.com/static/images/Fi/Final_16974958513780/Factors/1.png'
                    statname='Final_16974958513780/Factors'
                    path=''
                    ></VisualizeStat>
            </div>
            <hr/>
            <div className='m-5 d-flex flex-column'>
                <button className='d-flex px-3 align-items-center'>
                    <IonIcon icon={downloadOutline} className='image-fluid h1 pt-1 mx-3'></IonIcon>
                    <div className='w-100 text-start mx-1'>
                        Map of Maldives Waves | Maldives-Waves.pdf
                    </div>
                    <div>20/8/2023</div>
                </button>
                <button className='d-flex px-3 align-items-center'>
                    <IonIcon icon={downloadOutline} className='image-fluid h1 pt-1 mx-3'></IonIcon>
                    <div className='w-100 text-start mx-1'>
                        Red Algae Movement Data | RA_Movement.pdf
                    </div>
                    <div>1/12/2022</div>
                </button>
                <button className='d-flex px-3 align-items-center'>
                    <IonIcon icon={downloadOutline} className='image-fluid h1 pt-1 mx-3'></IonIcon>
                    <div className='w-100 text-start mx-1'>
                        Red Algae Zone in World | RA_WorldZone.png
                    </div>
                    <div>3/6/2022</div>
                </button>
                <button className='d-flex px-3 align-items-center'>
                    <IonIcon icon={downloadOutline} className='image-fluid h1 pt-1 mx-3'></IonIcon>
                    <div className='w-100 text-start mx-1'>
                        Maldives Ocean Dataset | Maldives-Ocean-Data.xlxs
                    </div>
                    <div>21/5/2022</div>
                </button>

            </div>
            <hr/>
            <div className='d-flex justify-content-center my-5'>
                <div style={{height: '175px'}} className='shadow w-75 d-flex px-5 py-4 align-items-center'>
                    <IonIcon icon={alertCircleOutline} className='display-1 '></IonIcon>
                    <div className='px-5 d-flex flex-column w-100 mt-4'>
                        <div className='h4'>
                            Subscribe to System Alert
                        </div>
                        <div>Get alerted via custom API, email, sms notification and other methods when certain events happen</div>
                        <button className='btn btn-success w-25 m-3 align-self-end'>
                            Subscribe
                        </button>
                    </div>
                </div>

            </div>
            <div className='m-5'></div>
        </div>
        
      </IonContent>
      
    </IonPage>
  );
};

export default StatHub;
