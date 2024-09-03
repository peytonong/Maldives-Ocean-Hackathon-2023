// About.tsx
import React from 'react';
import './About.css';
import {
  IonContent,
  IonPage,
  IonCard,
  IonCardHeader,
  IonCardSubtitle,
  IonCardTitle,
  IonCardContent,
  IonImg,
  IonText,
} from '@ionic/react';
import TopBar from '../components/topbar';
import member1 from '../assets/image/member1.jpg';
import member2 from '../assets/image/member2.jpg';
import member3 from '../assets/image/member3.jpg';
import member4 from '../assets/image/member4.jpg';
import member5 from '../assets/image/member5.jpg';

const About: React.FC = () => {
  return (
    <IonPage>
      <TopBar />
      <IonContent className="ion-padding">
        <div className='px-5'>
          
          
          <div className="head-section">
            <div className="objective-section text-center">
              <div className='display-5 m-3'>Our Objective</div>
              <p>
                Rhodo is a pioneering non-profit organization dedicated to developing predictive tools for forecasting red algae occurrences. Our mission is to raise awareness and provide advance notice to coastal communities, enabling them to prepare for potential red algae blooms. We are also committed to collecting important data from our users, including environmental factors that help assess water quality. Users can report their red algae observations, contributing to our comprehensive database. Additionally, our predictive model holds promise for future use in various applications beyond its initial purpose.
              </p>
            </div>
          </div>
          <div className='p-2'></div>
          <hr/>
          <div className='p-2'></div>

          <div className='display-5 m-3 text-center p-2'>Meet our team</div>

          {/* Team Members Section */}
          <div className="team-section">
            {renderTeamMember(member1, 'President of Phyta Foundation', 'Aasiya', 'Asia Pacific University(APU)', ' I am currently pursuing a Marketing degree at Asia Pacific University. My background in debate has honed my strong communication skills, which I am eager to utilize to effectively convey the importance of our predictive tools to a wider audience.')}
            {renderTeamMember(member2, 'Head of Collaboration and Innovation', 'Jessica', 'Asia Pacific University(APU)', ' As an Undergraduate in Industrial Design with a focus on Product Design at Asia Pacific University of Technology and Innovation, I bring a dedicated and detail-oriented approach to our team. I am driven to ensure that our predictive tools are not only effective but also user-friendly and aesthetically appealing.')}
            {renderTeamMember(member3, 'Head of Data Science and Research', 'Peython', 'Asia Pacific University(APU)', `Currently in my first year at Asia Pacific University of Technology & Innovation, I am pursuing a Bachelor's degree in Actuarial Science with a specialization in Financial Technology. My fascination with mathematics and data analytics fuels my commitment to our company's mission. I intend to leverage my skills to further our data collection and analysis efforts.`)}
            {renderTeamMember(member4, 'Head of Technology and Infrastructure', 'Brenden', 'Asia Pacific University(APU)', `My current course of study is Game Development at Asia Pacific University of Technology and Innovation. I have a deep passion for crafting immersive gaming experiences, and I leverage my skills in video creation, photo editing, and coding to bring our predictive tools to life. My creativity knows no bounds, and I'm excited to apply it to our projects.`)}
            {renderTeamMember(member5, 'Head of Research and Development', 'Chong', 'Asia Pacific University(APU)', `I am an IT enthusiast, actively working toward a degree in Information Technology at Asia Pacific University. Proficient in a multitude of programming languages and technologies, I am well-prepared to contribute to the development of our predictive tools. My primary interest lies in utilizing blockchain technology for secure data management, particularly in the context of international payments.`)}
          </div>
        </div>
      </IonContent>
    </IonPage>
  );
};

// Helper function to render team member cards
const renderTeamMember = (imageSrc: string, subtitle: string, title: string, text: string, content?: string) => (
  <IonCard className="container">
    <IonImg src={imageSrc} alt={title} className="card-img" />
    <IonCardHeader>
      <IonCardSubtitle>{subtitle}</IonCardSubtitle>
      <IonCardTitle>{title}</IonCardTitle>
      <IonText>{text}</IonText>
    </IonCardHeader>
    {content && <IonCardContent>{content}</IonCardContent>}
  </IonCard>
);

export default About;
