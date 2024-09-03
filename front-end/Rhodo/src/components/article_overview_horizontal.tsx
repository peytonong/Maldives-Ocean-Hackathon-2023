import React from 'react';
import { IonMenu, IonHeader, IonToolbar, IonTitle, IonContent, IonList, IonItem, IonIcon, IonLabel } from '@ionic/react';
import { libraryOutline, peopleCircleOutline, leafOutline, informationCircleOutline } from 'ionicons/icons';
import { useHistory } from 'react-router-dom';

interface iArticleOverview {
    logo: string,
    header: string,
    description: string,
    id: string
}

const ArticleOverviewHorizontal: React.FC<iArticleOverview> = (props) => {
    return(
        <button className='rounded-start shadow-sm p-3 d-flex bg-white align-items-center' 
        onClick={() => {
            window.location.href = 'community/article/' + props.id;
        }}
        style={{width:500, height: 175}}
        >
            <img src={props.logo} className='img-fluid w-25'></img>
            
            <div className='d-flex flex-column'>
                <b className='fw-bold h5 pt-1 px-3'>
                    {props.header}
                </b>
                <div className='content px-4'>
                <p className='inner text-start'>
                    {props.description}
                </p>
            </div>
            </div>
            
        </button>
    );

}

export default ArticleOverviewHorizontal;