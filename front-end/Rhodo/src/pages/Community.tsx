// Community.tsx
import React, { useState } from 'react';
import { IonButton, IonContent, IonModal, IonPage } from '@ionic/react';
import './Community.css';
import TopBar from '../components/topbar';
import communityImage from '../assets/image/community.jpg';
import Component from './Component';
import ArticleOverview from '../components/article_overview';
import ArticleOverviewHorizontal from '../components/article_overview_horizontal';
import { arrowForward } from 'ionicons/icons';

const Community: React.FC = () => {

  return (
    <IonPage>
      <TopBar></TopBar>
      <IonContent>
        <div className='px-5'>
          <div className='px-3 h1 mt-4'>Community</div>
          <div className='px-3'>Community driven space for researchers and alike to share and contribute</div>
          <hr/>
        </div>
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
      </IonContent>
    </IonPage>
  );
};

export default Community;
