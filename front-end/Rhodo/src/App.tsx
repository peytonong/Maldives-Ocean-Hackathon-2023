import { Redirect, Route } from 'react-router-dom';
// import {
//   IonApp,
//   IonContent,
//   IonIcon,
//   IonLabel,
//   IonRouterOutlet,
//   IonTabBar,
//   IonTabButton,
//   IonTabs,
//   setupIonicReact
// } from '@ionic/react';
import { IonReactRouter } from '@ionic/react-router';
import { ellipsisVerticalCircleOutline } from 'ionicons/icons';

/* Core CSS required for Ionic components to work properly */
import '@ionic/react/css/core.css';

/* Basic CSS for apps built with Ionic */
import '@ionic/react/css/normalize.css';
import '@ionic/react/css/structure.css';
import '@ionic/react/css/typography.css';

/* Optional CSS utils that can be commented out */
import '@ionic/react/css/padding.css';
import '@ionic/react/css/float-elements.css';
import '@ionic/react/css/text-alignment.css';
import '@ionic/react/css/text-transformation.css';
import '@ionic/react/css/flex-utils.css';
import '@ionic/react/css/display.css';

/* Theme variables */
import './theme/variables.css';
import 'bootstrap/dist/css/bootstrap.css';

/* Import Pages */
import TestHome from './pages/TestHome';
import Login from './pages/Login';
import Register from './pages/Register';
import About from './pages/About';
import Observation from './pages/Observation';
import NavBarMenu from './components/navbar_menu';
import Component from './pages/Component';
import MyPage from './pages/MyPage';
import Resource from './pages/Resource';
import Community from './pages/Community';
import Collaborate from './pages/Collaborate';
import ArticleContent from './pages/ArticleContent';
import Homepage from './pages/Homepage';
import KnowledgeHub from './pages/KnowledgeHub';
import DataFrame from './pages/Dataframe';
import StatHub from './pages/StatHub';

import { IonRouterOutlet, IonIcon, IonFab, IonApp, setupIonicReact, IonButtons, IonContent, IonHeader, IonMenu, IonMenuButton, IonPage, IonTitle, IonToolbar, IonSplitPane, IonList, IonFabButton} from '@ionic/react';




setupIonicReact();

const App: React.FC = () => (
  <IonApp>
    <IonSplitPane contentId="main-content" when="sm">
      <IonMenu contentId='main-content'>
        <NavBarMenu/>
      </IonMenu>
      <IonPage className='d-block' id="main-content">
        <IonFab className='d-block d-sm-none' slot="fixed" vertical="top" horizontal="start" > 
          <IonFabButton>
            <IonIcon icon={ellipsisVerticalCircleOutline}></IonIcon>
          </IonFabButton>
        </IonFab>
        <IonContent className='ion-page'>
          <IonReactRouter>
            <IonRouterOutlet>
              <Route path="/" component={Homepage}/>
              <Route path="/login" component={Login}/>
              <Route path="/register" component={Register}/>
              <Route path="/about" component={About}/>
              <Route path="/resource" component={Resource}/>
              <Route path="/resource/stathub" component={StatHub}/>
              <Route path="/resource/observation" component={Observation}/>
              <Route path="/resource/dataframe" component={DataFrame}/>
              <Route path="/component" component={Component}/>
              <Route path="/mypage" component={MyPage}/>
              <Route path="/community" component={Community}/>
              <Route path="/collaborate" component={Collaborate}/>
              <Route path="/articlecontent" component={ArticleContent}/>
              <Route path="/knowledgehub" component={KnowledgeHub}/>

            </IonRouterOutlet>
          </IonReactRouter>
        </IonContent>
      </IonPage>
    </IonSplitPane>
 
  </IonApp>
);

export default App;
