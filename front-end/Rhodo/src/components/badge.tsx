import { Component, h } from '@stencil/core';
import { IonHeader, IonToolbar, IonButtons, IonBackButton, IonTitle, IonContent, IonList, IonListHeader, IonItem, IonLabel, IonBadge, IonTabBar, IonTabButton, IonIcon } from '@ionic/react';


export class Badge {
  render() {
    return [
      <IonHeader translucent={true}>
        <IonToolbar>
          <IonButtons slot="start">
            <IonBackButton defaultHref="/"></IonBackButton>
          </IonButtons>
          <IonTitle>Badge</IonTitle>
        </IonToolbar>
      </IonHeader>,

      <IonContent fullscreen={true}>
        <IonList>
          <IonListHeader>
            <IonLabel>
              Badges
            </IonLabel>
          </IonListHeader>
          <IonItem>
            <IonLabel>Followers</IonLabel>
            <IonBadge slot="end">22k</IonBadge>
          </IonItem>
          <IonItem>
            <IonLabel>Likes</IonLabel>
            <IonBadge color="secondary" slot="end">118k</IonBadge>
          </IonItem>
          <IonItem>
            <IonLabel>Stars</IonLabel>
            <IonBadge color="tertiary" slot="end">34k</IonBadge>
          </IonItem>
          <IonItem>
            <IonLabel>Completed</IonLabel>
            <IonBadge color="success" slot="end">80</IonBadge>
          </IonItem>
          <IonItem>
            <IonLabel>Warnings</IonLabel>
            <IonBadge color="warning" slot="end">70</IonBadge>
          </IonItem>
          <IonItem>
            <IonLabel>Notifications</IonLabel>
            <IonBadge color="danger" slot="end">1000</IonBadge>
          </IonItem>
          <IonItem>
            <IonLabel>Unread</IonLabel>
            <IonBadge color="light" slot="end">24</IonBadge>
          </IonItem>
          <IonItem>
            <IonLabel>Drafts</IonLabel>
            <IonBadge color="medium" slot="end">14</IonBadge>
          </IonItem>
          <IonItem lines="full">
            <IonLabel>Deleted</IonLabel>
            <IonBadge color="dark" slot="end">4</IonBadge>
          </IonItem>
        </IonList>

        <IonTabBar>
          <IonTabButton selected>
            <IonIcon name="globe"></IonIcon>
            <IonBadge color="tertiary">44</IonBadge>
          </IonTabButton>
          <IonTabButton>
            <IonIcon name="people"></IonIcon>
            <IonBadge color="success">1</IonBadge>
          </IonTabButton>
          <IonTabButton>
            <IonIcon name="mail"></IonIcon>
            <IonBadge>2.3k</IonBadge>
          </IonTabButton>
        </IonTabBar>
      </IonContent>
    ];
  }
}
