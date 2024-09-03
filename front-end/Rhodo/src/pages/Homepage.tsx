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
import DummyStat from '../visualization/VisualizeStat';
import { libraryOutline } from 'ionicons/icons';
import VisualizeStat from '../visualization/VisualizeStat';
import AlgaePic from "../../public/alage.png";
import ArticleOverview from '../components/article_overview';
import ArticleOverviewHorizontal from '../components/article_overview_horizontal';

const Homepage: React.FC = () => {

    return (
        <IonPage>
            <TopBar></TopBar>
            <IonContent>
                <div className='d-flex justify-content-center'>
                <img src='../../public/Logo_Horizontal.png' width={300} height={300}></img>

                </div>
                <div className='d-flex flex-column mx-5 px-5 justify-content-center items-align-center text-center'>
                    <div className='display-6 mb-3'>
                        Maldives Community Initiative In Solving Harmful Algae Bloom (HAB)
                    </div>
                    <div className='h-5 mx-5 px-5'>
                        A collaboration of Maldivians driving a revolution in Marine data gathering in Maldives in an effort to combat Harmful Algae Bloom for the sustainability of Maldives.
                    </div>
                </div>
                <div className='my-5'></div>
                <div className='d-flex justify-content-center p-5 mx-5 shadow flex-column'>
                    <VisualizeStat width='1000px' height='450px' imgsrc='https://public.tableau.com/static/images/BX/BXHDNSGRJ/1_rss.png' 
                    staticimg='https://public.tableau.com/static/images/BX/BXHDNSGRJ/1.png'
                    statname='' path='shared/BXHDNSGRJ'></VisualizeStat>
                    <div className='text-end'>
                        See more <a href='/resource/stathub'>statistics and data {'>'}</a>
                    </div>
                </div>

                <div className='bg-image text-center shadow-1-strong rounded mb-5 text-white img-fluid mt-5'
                    style={{backgroundImage: 'url(../../public/red_tide_2.jpg)', width: '100%', height: 300 }}
                    >
                    <div className='d-flex flex-column p-5'>
                        <div className='mt-5 display-4'>Red Algae Bloom</div>
                        <div className='h6 p-5'>Red algae blooms, also known as red tides or harmful algal blooms, are natural phenomena that occur when certain species of microscopic algae, particularly dinoflagellates, undergo rapid and uncontrolled population growth.  </div>
                    </div>
                </div>

                <div className='d-flex justify-content-center p-5'>
                    <div className='shadow rounded d-flex justify-content-center flex-column align-items-center mx-3' style={{width: 300, height: 300}}>
                        {/* <IonIcon icon={libraryOutline} className='display-1 pb-6'></IonIcon> */}
                        <div className='h4 fw-bold'>$1 Billion in LOSS</div>
                        <div className='p-3'></div>
                        <div className='h6 text-center px-3'>US has spent $1 Billion in loss over the 10 years due to red algae and indicates the risk in East Asia</div>
                        <div className='blockquote-footer mt-3 text-end'>Applied Health Science</div>
                    </div>
                    <div className='shadow rounded d-flex justify-content-center flex-column align-items-center mx-3' style={{width: 300, height: 300}}>
                        {/* <IonIcon icon={libraryOutline} className='display-1 pb-6'></IonIcon> */}
                        <div className='h4 fw-bold text-center'>Disruption in the Marine ecosystem</div>
                        <div className='p-3'></div>
                        <div className='h6 text-center px-3'>Red Algae Blooms disrupts the marine ecosystem and contaminate the sea and the seafood</div>
                        {/* <div className='blockquote-footer mt-3 text-end'>Applied Health Science</div> */}
                    </div>
                    <div className='shadow rounded d-flex justify-content-center flex-column align-items-center mx-3' style={{width: 300, height: 300}}>
                        {/* <IonIcon icon={libraryOutline} className='display-1 pb-6'></IonIcon> */}
                        <div className='h4 fw-bold text-center'>Destroy's livelihood of Maldivians</div>
                        <div className='p-3'></div>
                        <div className='h6 text-center px-3'>Red Algae impacts the tourism and economy of Maldives and the health of the citizen</div>
                        {/* <div className='blockquote-footer mt-3 text-end'>Applied Health Science</div> */}
                    </div>
                </div>
                <div className='d-flex justify-content-end me-5 pe-5'>
                    <button className='btn btn-success me-5'
                     onClick={() => {
                        window.location.href = '/knowledgehub';
                      }}
                    >Learn More</button>
                </div>
                <div>
                <div className='px-5 mx-5'>
          <div className='display-6 pb-3'>Articles</div>
          <div className='d-flex'>
            <ArticleOverview logo='../../public/article_red_tide.jpg' header='Red Tide Concerns in Maldives' id='1'
              description=" 
              The Maldives, with its crystal-clear waters, pristine beaches, and vibrant marine life, is a tropical paradise that attracts tourists from all over the world. However, beneath this picturesque facade, the islands have had their share of environmental challenges, one of which is the occurrence of red tide. In this article, we'll delve into what red tide is, its impact on the Maldives, and the efforts being made to mitigate its effects.
              "></ArticleOverview>
              <div className='p-1'></div>
              <div className='d-flex flex-column'>
                <div className='py-3'></div>

                <ArticleOverviewHorizontal logo='../../public/sea.jpg' header='Glowing Maldives – Bioluminescence Beaches in The Maldives' id='1'
                description="
                Discover the enchanting phenomenon of bioluminescence in The Maldives, transforming its picturesque beaches into a magical spectacle at night. Beyond the sun-soaked images of golden shores and turquoise waters, this archipelago offers a unique experience with glowing beaches. Specialized Maldives tour packages unveil the mesmerizing glow created by bioluminescent organisms. 
                "></ArticleOverviewHorizontal>
                <div className='py-2'></div>
                <ArticleOverviewHorizontal logo='../../public/algal.png' header='Keep An Eye Out For The Deadly Algal Blooms Being Spotted Around The Maldives' id='1'
                description="
                Beware of Deadly Algal Blooms in the Maldives! The Maldives Marine Research Institute warns of potential algal blooms, rapid accumulations of algae in water systems identified by varying colors. These blooms can be deadly, producing toxins harmful to people, fish, and drinking water industries. The public is urged to report sightings for immediate action.
                "></ArticleOverviewHorizontal>
              </div>
          </div>
          <div className='d-flex justify-content-end py-5'>
            <button className='btn btn-success text-center'
             onClick={() => {
              window.location.href = 'community/article/';
              }}
            >View Articles {'>'}</button>          

          </div>
        </div>
        <div className='px-5 mx-5'>
                <div className='display-6 pb-3'>Events</div>
                <div className='d-flex'>
                    <ArticleOverview logo='../../public/campaign.jpg' header='Red Algae Awareness Campaign' id='1'
                    description=" 
                    Join us in spreading awareness about the impact of red algae on marine ecosystems. Learn about the causes, effects, and preventive measures to protect our oceans.
                    "></ArticleOverview>
                    <div className='p-1'></div>
                    <div className='d-flex flex-column'>
                        <div className='py-3'></div>

                        <ArticleOverviewHorizontal logo='../../public/clean.jpg' header='Beach Cleanup Day' id='1'
                        description="
                        Contribute to a cleaner environment by participating in our Beach Cleanup Day. Let's come together to remove plastic waste and debris, making our beaches healthier and more beautiful.
                        "></ArticleOverviewHorizontal>
                        <div className='py-2'></div>
                        <ArticleOverviewHorizontal logo='../../public/workshop.png' header='Community Workshop: Ocean Conservation' id='1'
                        description="
                        Join our workshop to understand the importance of ocean conservation. Discover ways to protect our oceans from pollution and preserve marine life for future generations.
                        "></ArticleOverviewHorizontal>
                    </div>
                </div>
                <div className='d-flex justify-content-end py-5'>
                    <button className='btn btn-success text-center'
                    onClick={() => {
                    window.location.href = 'community/events/';
                    }}
                    >View Events {'>'}</button>          
                </div>
                </div>
                </div>

            </IonContent>
        </IonPage>
    )
}

export default Homepage