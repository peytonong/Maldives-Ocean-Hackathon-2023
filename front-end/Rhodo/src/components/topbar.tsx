import React from 'react';
import { IonMenu, IonHeader, IonToolbar, IonTitle, IonContent, IonList, IonItem, IonIcon, IonLabel} from '@ionic/react';
import { personCircleOutline } from 'ionicons/icons';

const TopBar: React.FC = () => 
{
    return (
      <>
       <div className='container-fluid p-0'>
          <div className='BGBrandOffColor d-flex flex-row-reverse py-3 px-5'>
            <IonIcon icon={personCircleOutline} className='h1 text-white'></IonIcon>
          </div>
        </div>
      </>
   
    )    
};

export default TopBar;
