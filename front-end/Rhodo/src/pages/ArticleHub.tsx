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

const ArticleHub: React.FC = () => {

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

                <ArticleOverviewHorizontal logo='../../public/article_red_tide.jpg' header='Red Tide Concerns in Maldives' id='1'
                description="
                The Maldives, with its crystal-clear waters, pristine beaches, and vibrant marine life, is a tropical paradise that attracts tourists from all over the world. However, beneath this picturesque facade, the islands have had their share of environmental challenges, one of which is the occurrence of red tide. In this article, we'll delve into what red tide is, its impact on the Maldives, and the efforts being made to mitigate its effects.
                "></ArticleOverviewHorizontal>
                <div className='py-2'></div>
                <ArticleOverviewHorizontal logo='../../public/article_red_tide.jpg' header='Red Tide Concerns in Maldives' id='1'
                description="
                The Maldives, with its crystal-clear waters, pristine beaches, and vibrant marine life, is a tropical paradise that attracts tourists from all over the world. However, beneath this picturesque facade, the islands have had their share of environmental challenges, one of which is the occurrence of red tide. In this article, we'll delve into what red tide is, its impact on the Maldives, and the efforts being made to mitigate its effects.
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
            <ArticleOverview logo='../../public/events_awareness.jpg' header='Red Tide Concerns in Maldives' id='1'
              description=" 
              The Maldives, with its crystal-clear waters, pristine beaches, and vibrant marine life, is a tropical paradise that attracts tourists from all over the world. However, beneath this picturesque facade, the islands have had their share of environmental challenges, one of which is the occurrence of red tide. In this article, we'll delve into what red tide is, its impact on the Maldives, and the efforts being made to mitigate its effects.
              "></ArticleOverview>
              <div className='p-1'></div>
              <div className='d-flex flex-column'>
                <div className='py-3'></div>

                <ArticleOverviewHorizontal logo='../../public/events_awareness.jpg' header='Red Tide Concerns in Maldives' id='1'
                description="
                The Maldives, with its crystal-clear waters, pristine beaches, and vibrant marine life, is a tropical paradise that attracts tourists from all over the world. However, beneath this picturesque facade, the islands have had their share of environmental challenges, one of which is the occurrence of red tide. In this article, we'll delve into what red tide is, its impact on the Maldives, and the efforts being made to mitigate its effects.
                "></ArticleOverviewHorizontal>
                <div className='py-2'></div>
                <ArticleOverviewHorizontal logo='../../public/events_awareness.jpg' header='Red Tide Concerns in Maldives' id='1'
                description="
                The Maldives, with its crystal-clear waters, pristine beaches, and vibrant marine life, is a tropical paradise that attracts tourists from all over the world. However, beneath this picturesque facade, the islands have had their share of environmental challenges, one of which is the occurrence of red tide. In this article, we'll delve into what red tide is, its impact on the Maldives, and the efforts being made to mitigate its effects.
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

export default ArticleHub;
