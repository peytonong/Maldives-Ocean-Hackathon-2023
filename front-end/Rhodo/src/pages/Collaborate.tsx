import React, { useState } from 'react';
import { IonContent, IonModal, IonPage, IonToolbar } from '@ionic/react';
import ArticleContent from './ArticleContent';
import './Collaborate.css';
import TopBar from '../components/topbar';

// Assuming you have the company logos as imports
import company1Logo from '../../public/MSAzure.png';
import company2Logo from '../../public/MME.png';

// Define the FAQ questions and answers
const faqData = [
  { question: 'What is the collaboration process like?', answer: 'The collaboration process at Rhodo involves several key steps, including initial contact, preliminary meetings, alignment of goals, resource contribution, collaboration planning, data sharing and integration, and project implementation. This structured process ensures that partners and Rhodo work together seamlessly to achieve a better project outcome.' },
  { question: 'How can my company become a partner?', answer: 'If your company is interested in partnering with us, please contact us at main@rhodo.org, and our team will reach out to your organization shortly to gather more information and discuss potential collaboration opportunities.' },
  { question: 'Do you work with international businesses?', answer: 'Currently, we are focusing our efforts on the Maldives due to the specificity of the information and environmental factors involved. However, we have plans to expand our predictive model to other countries in the future. If you represent an international business interested in collaborating, please do contact us, and we will reach out to you once we successfully achieve this expansion. Together, we can make a global impact.' },
  // Add more FAQ items as needed
];

const Collaborate: React.FC = () => {
  const [isArticleModalOpen, setArticleModalOpen] = useState(false);
  const [faqVisibility, setFaqVisibility] = useState<Record<string, boolean>>({});

  const openArticleModal = () => {
    setArticleModalOpen(true);
  };

  const closeArticleModal = () => {
    setArticleModalOpen(false);
  };

  const toggleFaqVisibility = (questionKey: string) => {
    setFaqVisibility((prevVisibility) => ({
      ...prevVisibility,
      [questionKey]: !prevVisibility[questionKey],
    }));
  };

  return (
    <IonPage>
      <TopBar></TopBar>
      <IonContent>
        <div className='px-5'>
          <div className='py-4'></div>
          <div className="collaborate-header">
            <h2>Let's Join Forces with Rhodo for Red Algae Forecasting and Coastal Awareness</h2>
            <p>Collaborative Vision: Working Together for a Cleaner Coastline <br/>
                Open Doors: Welcoming Partnerships with Purpose<br/>
                Our Mission: Predicting Red Algae, Protecting Coastal Communities</p>

          {/* Meet Our Partners Content */}
          <div className="collaborate-container">
            <h3>Meet Our Partners</h3>
            <p>Collectively, we are earnestly striving to safeguard our invaluable coastlines. Allow us to introduce the remarkable allies supporting our cause:</p>
          </div>

          {/* Company Logos Section */}
          <div className="company-logos">
            <img src={company1Logo} alt="Company 1 Logo" />
            <img src={company2Logo} alt="Company 2 Logo" />
          </div>
          </div>

          {/* Frequently Asked Questions Section */}
          <div className="faq-section">
            <h3>Frequently Asked Questions</h3>
            {faqData.map((faqItem) => (
              <div className="faq-item" key={faqItem.question}>
                <div className="question" onClick={() => toggleFaqVisibility(faqItem.question)}>
                  <h4>{faqItem.question}</h4>
                  <span className={faqVisibility[faqItem.question] ? 'arrow up' : 'arrow down'}>&#9660;</span>
                </div>
                {faqVisibility[faqItem.question] && <p>{faqItem.answer}</p>}
              </div>
            ))}
          </div>

          {/* Article Modal */}
          <IonModal isOpen={isArticleModalOpen}>
            <div className="modal">
              <div className="modal-content">
                <span className="close" onClick={closeArticleModal}>
                  &times;
                </span>
                {/* Render the ArticleContent component */}
                <ArticleContent />
              </div>
            </div>

            {/* Company Logos Section */}
            <div className="company-logos">
              <img src={company1Logo} alt="Company 1 Logo" />
              <img src={company2Logo} alt="Company 2 Logo" />
            </div>

            {/* Frequently Asked Questions Section */}
            <div className="faq-section">
              <h3>Frequently Asked Questions</h3>
              {faqData.map((faqItem) => (
                <div className="faq-item" key={faqItem.question}>
                  <div className="question" onClick={() => toggleFaqVisibility(faqItem.question)}>
                    <h4>{faqItem.question}</h4>
                    <span className={faqVisibility[faqItem.question] ? 'arrow up' : 'arrow down'}>&#9660;</span>
                  </div>
                  {faqVisibility[faqItem.question] && <p>{faqItem.answer}</p>}
                </div>
              ))}
            </div>

            {/* Article Modal */}
            <IonModal isOpen={isArticleModalOpen}>
              <div className="modal">
                <div className="modal-content">
                  <span className="close" onClick={closeArticleModal}>
                    &times;
                  </span>
                  {/* Render the ArticleContent component */}
                  <ArticleContent />
                </div>
              </div>
            </IonModal>
          </IonModal>
        </div>
        
      </IonContent>
    </IonPage>
  ); 
};

export default Collaborate;
