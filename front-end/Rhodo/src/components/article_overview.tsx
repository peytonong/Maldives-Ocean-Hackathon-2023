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

const ArticleOverview: React.FC<iArticleOverview> = (props) => {
    return(
        <button className='rounded-start shadow-sm p-3 d-flex flex-column w-50 bg-white' 
            onClick={() => {
                window.location.href = 'community/article/' + props.id;
            }}>
            <img src={props.logo} className='img-fluid p-3'></img>
            <b className='fw-bold h5 pt-1 px-3'>
                {props.header}
            </b>
            <div className='content px-3'>
                <p className='inner te'>
                    {props.description}
                </p>
            </div>
        </button>
    );

}

export default ArticleOverview;