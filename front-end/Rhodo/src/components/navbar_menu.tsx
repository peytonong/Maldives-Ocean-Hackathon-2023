import React from 'react';
import { IonMenu, IonHeader, IonToolbar, IonTitle, IonContent, IonList, IonItem, IonIcon, IonLabel } from '@ionic/react';
import { homeOutline, schoolOutline, libraryOutline, peopleCircleOutline, leafOutline, informationCircleOutline } from 'ionicons/icons';
import { useHistory } from 'react-router-dom';


const NavBarMenu: React.FC = () => 
{
    const history = useHistory();
    return (
    <>
    <IonContent>
        <div className='d-flex'>
            <div className='d-flex justify-content-center align-items-center w-50 mx-auto my-3'>
                <a href='/'>
                    <img src="../../public/Logo_Vertical.png" className='img-fluid' onClick={() => history.push('/about-us')}></img>
                </a>
            </div>
        </div>
        <div>
            <IonList >
            <IonItem button routerLink="/">
            <IonIcon icon={homeOutline} slot="start" />
            <IonLabel>Home</IonLabel>
            </IonItem>
            <IonItem button routerLink="/about">
            <IonIcon icon={informationCircleOutline} slot="start" />
            <IonLabel>About Us</IonLabel>
            </IonItem>
            <IonItem button routerLink="/resource">
            <IonIcon icon={libraryOutline} slot="start" />
            <IonLabel>Resource</IonLabel>
            </IonItem>
            <IonItem button routerLink="/community">
            <IonIcon icon={peopleCircleOutline} slot="start" />
            <IonLabel>Community</IonLabel>
            </IonItem>
            <IonItem button routerLink="/collaborate">
            <IonIcon icon={leafOutline} slot="start" />
            <IonLabel>Collaborate</IonLabel>
            </IonItem>
            <IonItem button routerLink="/knowledgehub">
            <IonIcon icon={schoolOutline} slot="start" />
            <IonLabel>Knowledge Hub</IonLabel>
            </IonItem>
            </IonList>
        </div>
        </IonContent>
      
    </>
    )    
};

export default NavBarMenu;
