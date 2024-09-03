// KnowledgeHub.tsx

import React, { useState } from 'react';
import {
  IonContent,
  IonHeader,
  IonPage,
  IonTitle,
  IonToolbar,
  IonCard,
  IonCardHeader,
  IonCardTitle,
  IonCardContent,
  IonGrid,
  IonRow,
  IonCol,
} from '@ionic/react';
import './KnowledgeHub.css';
import TopBar from '../components/topbar';

const KnowledgeHub: React.FC = () => {
  const knowledgeData = [
    { id: 1, title: 'What are red algae?', content: 'Red algae are the oldest group of eukaryotic algae containing over 6000 species. They fall under the kingdom Protista and phylum Rhodophyta. They contain chlorophyll and can prepare their own food by the process of photosynthesis.' },
    { id: 2, title: 'What is the importance of red algae?', content: 'Red algae form an important part of the ecosystem and are consumed by various organisms such as crustaceans, fish, worms and even humans. Red algae are also used to produce agar that is used as a food additive. They are rich in calcium and also used in vitamin supplements.' },
    { id: 3, title: 'Where are red algae found?', content: 'Red algae are commonly found in coral reefs and tide pools. They have the ability to survive at a greater depth than other algae because the pigment Phycoerythrin absorbs the blue light that can penetrate deeper than any other light wave. This allows red algae to carry out photosynthesis at a greater depth.' },
    { id: 4, title: 'What distinguishes red algae from other algae?', content: 'The only difference between the red algae and other algae is that the red algae lack flagella, the whip-like structures that help in locomotion and perform sensory functions.' },
    // Add more knowledge items as needed
  ];

  // State to track the expanded state for each card
  const [expandedCards, setExpandedCards] = useState<number[]>([]);

  const handleKnowledgeClick = (id: number) => {
    // Toggle the expanded state for the clicked card
    setExpandedCards((prevExpanded) => {
      if (prevExpanded.includes(id)) {
        // If the card is already expanded, remove it from the array
        return prevExpanded.filter((cardId) => cardId !== id);
      } else {
        // If the card is not expanded, add it to the array
        return [...prevExpanded, id];
      }
    });
  };

  return (
    <IonPage>
      <TopBar></TopBar>

      <IonContent>
        <div className='px-5'>
          <IonGrid>
            
            {/* Heading and paragraphs */}
            <IonRow className="heading-section">
              <IonCol>
                <h1>Welcome to the Knowledge Hub</h1>
                <p>
                  Explore and learn more about various topics in our knowledge base.
                </p>
                <p>
                  Click on a knowledge item to view its content.
                </p>
              </IonCol>
            </IonRow>

            {/* Card containers */}
            <IonRow className="knowledge-cards-section">
              {knowledgeData.map((knowledge) => (
                <IonCol size="6" key={knowledge.id}>
                  <IonCard
                    className={`knowledge-card ${expandedCards.includes(knowledge.id) ? 'expanded' : ''}`}
                    onClick={() => handleKnowledgeClick(knowledge.id)}
                  >
                    <IonCardHeader>
                      <IonCardTitle>{knowledge.title}</IonCardTitle>
                    </IonCardHeader>
                    <IonCardContent className="card-content">
                      {knowledge.content}
                    </IonCardContent>
                  </IonCard>
                </IonCol>
              ))}
            </IonRow>

            {/* Overview section */}
            <IonRow className="overview-section">
              <IonCol>
                <h1>Overview of Red Algae</h1>
                <p>
                  Red algae, scientifically classified under the kingdom Protista and phylum Rhodophyta, constitute one of the oldest groups of eukaryotic algae. With over 6000 species, red algae play a vital role in various ecosystems and human activities.
                </p>
                <p>
                  These algae are distinguishable by their red pigmentation, caused by the presence of the pigment phycoerythrin. Red algae are particularly interesting for their unique features, ecological significance, and various uses in different industries.
                </p>
              </IonCol>
            </IonRow>

          </IonGrid>            
        </div>
      </IonContent>
    </IonPage>
  );
};

export default KnowledgeHub;
