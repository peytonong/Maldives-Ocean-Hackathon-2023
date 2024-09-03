#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os,time
from influxdb_client_3 import InfluxDBClient3, Point
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report, roc_curve,roc_auc_score
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import statistics as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report, roc_curve,roc_auc_score
import statistics as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

##import csv data & data cleaning

df1 = pd.read_csv('AlgaeOcr2.csv')
df1["date"] = pd.to_datetime(df1["date"],format="%d/%m/%Y")
df1.sort_values(by="date")
df1 = df1.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df1 = df1.dropna(subset=["date","temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
df1 = df1.drop_duplicates()
num = ["temperature","ph","salinity",'dis_oxy',"light","water_speed","nutrient","nutrient2"]
for k in num:
    Q1 = df1[k].quantile(0.25)
    Q3 = df1[k].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.0 * IQR
    upper_bound = Q3 + 2.0 * IQR
    df1 = df1[(df1[k] >= lower_bound) & (df1[k] <= upper_bound)]
print("Ocr: \n",df1.head(2),"\n")

df2 = pd.read_csv('AlgaeSize2.csv')
df2["date"] = pd.to_datetime(df2["date"],format="%d/%m/%Y")
df2.sort_values(by="date")
df2 = df2.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df2 = df2.dropna(subset=["date","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2","size"])
df2 = df2.drop_duplicates()
num = ["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2","size"]
for k in num: 
    Q1 = df2[k].quantile(0.25)
    Q3 = df2[k].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.0 * IQR
    upper_bound = Q3 + 2.0 * IQR
    df2 = df2[(df2[k] >= lower_bound) & (df2[k] <= upper_bound)]
print("Size: \n",df2.head(2),"\n")


# In[ ]:


#import data to InfluxDB

token = "bIq_bcybf3PNahATbM75_rx2-koAqxmMypxSjVbmy80NXzK5eXaeXGYRttGmzdN1KdsEtdR0c-QLPNGHAJ9c-Q=="
org = "None"
host = "https://us-east-1-1.aws.cloud2.influxdata.com"
database ="Maldives OH 2023"
client = InfluxDBClient3(host=host, token=token, org=org)

##### USE THIS TO DELETE DATA IF ACCIDENTALLY INPUT TWICE
# client.delete_api().delete(
#     predicate =' _measurement == "Test"',
#     start=datetime(1970,1,1),
#     stop=datetime(2100,1,1),
#     bucket=database,
#     org=org
# )

##### ONLY COULD RUN ONE TIME TO INITIATE THE DATASET IMPORT
# df1 = df1.to_json(orient="records")
# df1j = json.loads(df1j)
# for key in df1j:
#     point = Point("Test")
#     point.field("date",key["date"])
#     point.tag("latitude", key["latitude"])
#     point.tag("longitude",key["longtitude"])
#     point.field("temperature", key["temperature"])
#     point.field("ph", key["ph"])
#     point.field("salinity",key["salinity"])
#     point.field("dis_oxy",key["dis_oxy"])
#     point.field("light",key["light"])
#     point.field("water_speed",key["water_speed"])
#     point.field("nutrient", key["nutrient"])
#     point.field("nutrient2",key["nutrient2"])
#     point.field("occurrence", key["occurrence"])
#     client.write(database=database, record=point)
# print("complete")


# In[ ]:


#import data back to jupyter from influxdb
token = "bIq_bcybf3PNahATbM75_rx2-koAqxmMypxSjVbmy80NXzK5eXaeXGYRttGmzdN1KdsEtdR0c-QLPNGHAJ9c-Q=="
org = "None"
host = "https://us-east-1-1.aws.cloud2.influxdata.com"
database="Maldives OH 2023"
client = InfluxDBClient3(host=host, token=token, org=org)

#import OCCR
query = """SELECT *
FROM 'OCCR'
WHERE time >= now() - interval '24 hours'
AND ('bees' IS NOT NULL OR 'ants' IS NOT NULL)"""
table = client.query(query=query, database="Maldives OH 2023", language='sql') 
df1 = table.to_pandas().sort_values(by="date")
print(df1.head(2))

#import SIZE
query = """SELECT *
FROM 'SIZE'
WHERE time >= now() - interval '24 hours'
AND ('bees' IS NOT NULL OR 'ants' IS NOT NULL)"""
table = client.query(query=query, database="Maldives OH 2023", language='sql') 
df2 = table.to_pandas().sort_values(by="date")
print(df2.head(2))


# In[ ]:


#Grouping
df1['lat_rad'] = np.radians(df1['latitude'])
df1['lon_rad'] = np.radians(df1['longtitude'])
X = df1[['lat_rad', 'lon_rad']]

k = 2
kmeans = KMeans(n_clusters=k,n_init=10)
df1['cluster'] = kmeans.fit_predict(X)

#for illustration purpose
for cluster_id in range(k):
    plt.scatter(df1[df1['cluster'] == cluster_id]['longtitude'], 
                df1[df1['cluster'] == cluster_id]['latitude'], 
                label=f'Cluster {cluster_id}')
plt.legend()
plt.xlabel('Longtitude')
plt.ylabel('Latitude')
plt.show()


# In[ ]:


#Evaluate the performance of each model

dfl=[]
for k in df1["cluster"].unique():
    df = df1.loc[df1["cluster"]==k,]
    print(df["occurrence"].unique())
    dfl.append(df)
    
accuracys_gbc=[]
accuracys_dt=[]
accuracys_rf=[]
accuracys_lr=[]
accuracys_svm=[]

gbcm=[]
dtm=[]
rfm=[]
lrm=[]
svm=[]

for dfk in dfl:
    dfk=dfk.sort_values(by="date",ascending=False)
    X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]
    X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
    Y = dfk["occurrence"]

    train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                        test_size = 0.25, 
                                                        random_state = 123)

    gbc = GradientBoostingClassifier(n_estimators=1000,
                                     learning_rate=0.05,
                                     random_state=100,
                                     max_features=5 )
    gbcm.append(gbc)
    gbc.fit(train_X, train_y)
    pred_y = gbc.predict(test_X)
    acc = accuracy_score(test_y, pred_y)
    accuracys_gbc.append(acc)

    d=DecisionTreeClassifier()
    d.fit(train_X,train_y)
    dtm.append(d)
    pred_y=d.predict(test_X)
    acc = accuracy_score(test_y, pred_y)
    accuracys_dt.append(acc)

    rf=RandomForestClassifier()
    rf.fit(train_X,train_y)
    rfm.append(rf)
    pred_y=rf.predict(test_X)
    acc = accuracy_score(test_y, pred_y)
    accuracys_rf.append(acc)

    lr=LogisticRegression(solver='lbfgs',max_iter=1000)
    lr.fit(train_X,train_y)
    lrm.append(lr)
    pred_y=lr.predict(test_X)
    acc = accuracy_score(test_y, pred_y)
    accuracys_lr.append(acc)

    svm=SVC(kernel="rbf",degree=3,C=1.0,gamma=0.009)
    svm.fit(train_X,train_y)
    pred_y=svm.predict(test_X)
    acc = accuracy_score(test_y, pred_y)
    accuracys_svm.append(acc)

print(accuracys_gbc,"\n average of gbc: ",st.mean(accuracys_gbc),"\n")
print(accuracys_dt,"\n average of dt: ",st.mean(accuracys_dt),"\n")
print(accuracys_rf,"\n average of rf: ",st.mean(accuracys_rf),"\n")
print(accuracys_lr,"\n average of lr: ",st.mean(accuracys_lr),"\n")
print(accuracys_svm,"\n average of svm: ",st.mean(accuracys_svm),"\n")

#SELECT BEST MODEL AND USE IT LATER


# In[ ]:


##LSTM FOR EACH FACTOR

final=pd.DataFrame()
dates=pd.to_datetime(["2022/01/01","2023/01/01","2024/01/01","2025/01/01","2026/01/01","2027/01/01","2028/01/01","2029/01/01","2030/01/01","2031/01/01"],format="%Y/%m/%d")

dfk=df1.loc[df1["cluster"]==0]
pred1=pd.DataFrame()
factors=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]
for k in factors:
    sequence_length = 10
    input_sequences = []
    target_sequences = []
    for i in range(len(dfk) - sequence_length):
        X_seq = dfk[k].values[i]
        Y_seq = dfk[k].values[i+sequence_length]
        X_dates = dfk[k].values[i]
        input_sequences.append({"dates":X_dates,"input":X_seq})
        target_sequences.append(dfk.iloc[i+sequence_length,9])

    X_sequences_df = pd.DataFrame(input_sequences)
    Y_sequences_df = pd.Series(target_sequences, name='target')

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_sequences_df, Y_sequences_df, test_size=0.2, shuffle=False,random_state=20)

    model = Sequential()
    model.add(LSTM(50, input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error') 
    model.fit(X_train["input"], Y_train, epochs=20, batch_size=32) 
    initial_sequence = dfk[k].values[-sequence_length:] 
    predicted_data = list(initial_sequence)
    num_steps_to_predict = 10

    for step in range(num_steps_to_predict):
        sequence_length=len(initial_sequence)
        input_data = np.array(initial_sequence).reshape(1,sequence_length, 1)
        next_data_point = model.predict(input_data)
        predicted_data.append(next_data_point[0, 0])
        initial_sequence = initial_sequence[1:] + [next_data_point[0, 0]]
    predicted_data = predicted_data[:-num_steps_to_predict]
    last10 = predicted_data[-num_steps_to_predict:]
    pred1[k]=last10
X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]
X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
Y = dfk["occurrence"]
train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size = 0.25, 
                                                    random_state = 123)
rm = RandomForestClassifier()
rm.fit(train_X, train_y)
pred_y = rm.predict(pred1)
pred1O=pred_y

X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]
X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
Y = dfk["longtitude"]
train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size = 0.25, 
                                                    random_state = 123)
md = LinearRegression()
md.fit(train_X, train_y)
pred_y = md.predict(pred1)
pred1L=pred_y

X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]
X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
Y = dfk["latitude"]
train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size = 0.25, 
                                                    random_state = 123)
md2 = LinearRegression()
md2.fit(train_X, train_y)
pred_y = md2.predict(pred1)
pred1Lt=pred_y

pred1["date"]=dates
pred1["cluster"]=np.repeat(0,10)
pred1["occurrence"]=pred1O
pred1["longtitude"]=pred1L
pred1["latitude"]=pred1Lt

df2m=[]
dfk=df1.loc[df1["cluster"]==1]
pred2=pd.DataFrame()
factors=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]
for k in factors:
    sequence_length = 10
    input_sequences = []
    target_sequences = []
    for i in range(len(dfk) - sequence_length):
        X_seq = dfk[k].values[i]
        Y_seq = dfk[k].values[i+sequence_length]
        X_dates = dfk[k].values[i]
        input_sequences.append({"dates":X_dates,"input":X_seq})
        target_sequences.append(dfk.iloc[i+sequence_length,9])

    X_sequences_df = pd.DataFrame(input_sequences)
    Y_sequences_df = pd.Series(target_sequences, name='target')

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_sequences_df, Y_sequences_df, test_size=0.2, shuffle=False,random_state=20)

    model = Sequential()
    model.add(LSTM(50, input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error') 
    model.fit(X_train["input"], Y_train, epochs=20, batch_size=32) 

    initial_sequence = dfk[k].values[-sequence_length:] 
    predicted_data = list(initial_sequence)
    num_steps_to_predict = 10

    for step in range(num_steps_to_predict):
        sequence_length=len(initial_sequence)
        input_data = np.array(initial_sequence).reshape(1,sequence_length, 1)
        next_data_point = model.predict(input_data)
        predicted_data.append(next_data_point[0, 0])
        initial_sequence = initial_sequence[1:] + [next_data_point[0, 0]]
    predicted_data = predicted_data[:-num_steps_to_predict]
    last10 = predicted_data[-num_steps_to_predict:]
    pred2[k]=last10
X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]
X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
Y = dfk["occurrence"]
train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size = 0.25, 
                                                    random_state = 123)
rm = RandomForestClassifier()
rm.fit(train_X, train_y)
pred_y = rm.predict(pred2)
pred2O=pred_y

X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]
X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
Y = dfk["longtitude"]
train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size = 0.25, 
                                                    random_state = 123)
md = LinearRegression()
md.fit(train_X, train_y)
pred_y = md.predict(pred2)
pred2L=pred_y

X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]
X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
Y = dfk["latitude"]
train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size = 0.25, 
                                                    random_state = 123)
md2 = LinearRegression()
md2.fit(train_X, train_y)
pred_y = md2.predict(pred2)
pred2Lt=pred_y

X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]

pred2["date"]=dates
pred2["cluster"]=np.repeat(1,10)
pred2["occurrence"]=pred2O
pred2["longtitude"]=pred2L
pred2["latitude"]=pred2Lt

final = pd.concat([pred1,pred2,final])
print(final)
## FINAL -> ALL PREDICTION OF ALL Y FOR 10 STEPS


# In[ ]:


##smoothing before export data

#smooth final
final=final.sort_values(by="date")
final=final.sort_values(by="longtitude")
final=final.sort_values(by="latitude")
def moving_average_smoothing(X,k):
    S = np.zeros(X.shape[0])
    for t in range(X.shape[0]):
        if t < k:
            S[t] = np.mean(X[:t+1])
        else:
            S[t] = np.sum(X[t-k:t])/k
    return S
final["latitude"]=pd.DataFrame(moving_average_smoothing(np.array(final["latitude"]),4))
final["longtitude"]=pd.DataFrame(moving_average_smoothing(np.array(final["longtitude"]),5))

#smooth df1
df1=df1.sort_values(by="date")
df1=df1.sort_values(by="longtitude")
df1=df1.sort_values(by="latitude")
def moving_average_smoothing(X,k):
    S = np.zeros(X.shape[0])
    for t in range(X.shape[0]):
        if t < k:
            S[t] = np.mean(X[:t+1])
        else:
            S[t] = np.sum(X[t-k:t])/k
    return S
df1["latitude"]=pd.DataFrame(moving_average_smoothing(np.array(df1["latitude"]),8))
df1["longtitude"]=pd.DataFrame(moving_average_smoothing(np.array(df1["longtitude"]),12))

final["type"]="prediction"
df1["type"]="historical data"
combine = pd.concat([final,df1])
print(combine.head(2))
plt.scatter(combine[combine['cluster'] == cluster_id]['longtitude'], 
             combine[combine['cluster'] == cluster_id]['latitude'])


# In[ ]:


# Import back to InfluxDB

token = "bIq_bcybf3PNahATbM75_rx2-koAqxmMypxSjVbmy80NXzK5eXaeXGYRttGmzdN1KdsEtdR0c-QLPNGHAJ9c-Q=="
org = "None"
host = "https://us-east-1-1.aws.cloud2.influxdata.com"
database ="Maldives OH 2023"
client = InfluxDBClient3(host=host, token=token, org=org)

##### USE THIS TO DELETE DATA IF ACCIDENTALLY INPUT TWICE
# client.delete_api().delete(
#     predicate =' _measurement == "Test"',
#     start=datetime(1970,1,1),
#     stop=datetime(2100,1,1),
#     bucket=database,
#     org=org
# )

##### ONLY COULD RUN ONE TIME TO INITIATE THE DATASET IMPORT
# combine = combine.to_json(orient="records")
# combinej = json.loads(combinej)
# for key in combinej:
#     point = Point("Test")
#     point.field("date",key["date"])
#     point.tag("latitude", key["latitude"])
#     point.tag("longitude",key["longtitude"])
#     point.field("temperature", key["temperature"])
#     point.field("ph", key["ph"])
#     point.field("salinity",key["salinity"])
#     point.field("dis_oxy",key["dis_oxy"])
#     point.field("light",key["light"])
#     point.field("water_speed",key["water_speed"])
#     point.field("nutrient", key["nutrient"])
#     point.field("nutrient2",key["nutrient2"])
#     point.field("occurrence", key["occurrence"])
#     point.field("type",key["type"])
#     client.write(database=database, record=point)
# print("complete")


# In[ ]:


#DF2 SIZE

X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]
X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
Y = dfk["size"]
train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size = 0.25, 
                                                    random_state = 123)
md2 = LinearRegression()
md2.fit(train_X, train_y)
pred_y = md2.predict(pred1)

X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
Y = dfk["size"]
train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size = 0.25, 
                                                    random_state = 123)
md2 = LinearRegression()
md2.fit(train_X, train_y)
pred_y = md2.predict(pred2)
pred1S=pred_y

final["type"]="prediction"
df1["type"]="original"
combineS = pd.concat([final,df1])
print(combineS.head(2))


# In[ ]:


# Import back to InfluxDB

token = "bIq_bcybf3PNahATbM75_rx2-koAqxmMypxSjVbmy80NXzK5eXaeXGYRttGmzdN1KdsEtdR0c-QLPNGHAJ9c-Q=="
org = "None"
host = "https://us-east-1-1.aws.cloud2.influxdata.com"
database ="Maldives OH 2023"
client = InfluxDBClient3(host=host, token=token, org=org)

##### USE THIS TO DELETE DATA IF ACCIDENTALLY INPUT TWICE
# client.delete_api().delete(
#     predicate =' _measurement == "Test"',
#     start=datetime(1970,1,1),
#     stop=datetime(2100,1,1),
#     bucket=database,
#     org=org
# )

##### ONLY COULD RUN ONE TIME TO INITIATE THE DATASET IMPORT
# combineS = combineS.to_json(orient="records")
# combineSj = json.loads(combineSj)
# for key in combineSj:
#     point = Point("Test")
#     point.field("date",key["date"])
#     point.tag("latitude", key["latitude"])
#     point.tag("longitude",key["longtitude"])
#     point.field("temperature", key["temperature"])
#     point.field("ph", key["ph"])
#     point.field("salinity",key["salinity"])
#     point.field("dis_oxy",key["dis_oxy"])
#     point.field("light",key["light"])
#     point.field("water_speed",key["water_speed"])
#     point.field("nutrient", key["nutrient"])
#     point.field("nutrient2",key["nutrient2"])
#     point.field("occurrence", key["occurrence"])
#     point.field("type",key["type"])
#     point.filed("size",key["size"])
#     client.write(database=database, record=point)
# print("complete")


# In[ ]:


## grouping

df2['lat_rad'] = np.radians(df2['latitude'])
df2['lon_rad'] = np.radians(df2['longtitude'])
X = df2[['lat_rad', 'lon_rad']]

k = 2
kmeans = KMeans(n_clusters=k,n_init=10)
df2['cluster'] = kmeans.fit_predict(X)

# import matplotlib.pyplot as plt

# for cluster_id in range(k):
#     plt.scatter(df2[df2['cluster'] == cluster_id]['longtitude'], 
#                 df2[df2['cluster'] == cluster_id]['latitude'], 
#                 label=f'Cluster {cluster_id}')
# plt.legend()
# plt.xlabel('Longtitude')
# plt.ylabel('Latitude')
# plt.show()


# In[ ]:


df2a = df2.loc[df2["cluster"]==0,]
df2b = df2.loc[df2["cluster"]==1,]
dfl2=[df2a,df2b]


# In[ ]:


## find best model

mse_lrg=[]
r2_lrg=[]

dfk=df2a
X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]
X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
Y = dfk["size"]
train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size = 0.25, 
                                                    random_state = 123)
model = LinearRegression()
model.fit(train_X, train_y)
Y_pred = model.predict(test_X)
mse = mean_squared_error(test_y, Y_pred)
mse_lrg.append(mse)
r_squared = r2_score(test_y, Y_pred)
if(math.isnan(r_squared)):
    r_squared=0
    r2_lrg.append(r_squared)
else:
    r2_lrg.append(r_squared)

print(mse_lrg,"\n 1 average of mse of lrg: ",st.mean(mse_lrg),"\n")
print(r2_lrg,"\n 1 average of r2 of lrg: ",st.mean(r2_lrg),"\n")

mse_lrg=[]
r2_lrg=[]

dfk=df2b
X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]
X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
Y = dfk["size"]
train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size = 0.25, 
                                                    random_state = 123)
model = LinearRegression()
model.fit(train_X, train_y)
Y_pred = model.predict(test_X)
mse = mean_squared_error(test_y, Y_pred)
mse_lrg.append(mse)
r_squared = r2_score(test_y, Y_pred)
if(math.isnan(r_squared)):
    r_squared=0
    r2_lrg.append(r_squared)
else:
    r2_lrg.append(r_squared)

print(mse_lrg,"\n 2 average of mse of lrg: ",st.mean(mse_lrg),"\n")
print(r2_lrg,"\n 2 average of r2 of lrg: ",st.mean(r2_lrg),"\n")


# In[ ]:


## LSTM FOR EACH FACTORS

finals=pd.DataFrame()
dates=pd.to_datetime(["2022/01/01","2023/01/01","2024/01/01","2025/01/01","2026/01/01","2027/01/01","2028/01/01","2029/01/01","2030/01/01","2031/01/01"],format="%Y/%m/%d")

dfk=df2a
pred1=pd.DataFrame()
factors=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]
for k in factors:
    sequence_length = 10
    input_sequences = []
    target_sequences = []
    for i in range(len(dfk) - sequence_length):
        X_seq = dfk[k].values[i]
        Y_seq = dfk[k].values[i+sequence_length]
        X_dates = dfk[k].values[i]
        input_sequences.append({"dates":X_dates,"input":X_seq})
        target_sequences.append(dfk.iloc[i+sequence_length,9])

    X_sequences_df = pd.DataFrame(input_sequences)
    Y_sequences_df = pd.Series(target_sequences, name='target')

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_sequences_df, Y_sequences_df, test_size=0.2, shuffle=False,random_state=20)

    model = Sequential()
    model.add(LSTM(50, input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error') 
    model.fit(X_train["input"], Y_train, epochs=20, batch_size=32) 
    initial_sequence = dfk[k].values[-sequence_length:] 
    predicted_data = list(initial_sequence)
    num_steps_to_predict = 10

    for step in range(num_steps_to_predict):
        sequence_length=len(initial_sequence)
        input_data = np.array(initial_sequence).reshape(1,sequence_length, 1)
        next_data_point = model.predict(input_data)
        predicted_data.append(next_data_point[0, 0])
        initial_sequence = initial_sequence[1:] + [next_data_point[0, 0]]
    predicted_data = predicted_data[:-num_steps_to_predict]
    last10 = predicted_data[-num_steps_to_predict:]
    pred1[k]=last10
X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]
X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
Y = dfk["size"]
train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size = 0.25, 
                                                    random_state = 123)
model = LinearRegression()
model.fit(train_X, train_y)
pred_y = model.predict(pred1)
pred1["size"]=pred_y
pred1["date"]=dates
pred1["cluster"]=np.repeat(0,10)
finals = pd.concat([pred1,finals])

df1m=[]
dfk=df2b
pred2=pd.DataFrame()
factors=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]
for k in factors:
    sequence_length = 10
    input_sequences = []
    target_sequences = []
    for i in range(len(dfk) - sequence_length):
        X_seq = dfk[k].values[i]
        Y_seq = dfk[k].values[i+sequence_length]
        X_dates = dfk[k].values[i]
        input_sequences.append({"dates":X_dates,"input":X_seq})
        target_sequences.append(dfk.iloc[i+sequence_length,9])

    X_sequences_df = pd.DataFrame(input_sequences)
    Y_sequences_df = pd.Series(target_sequences, name='target')

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_sequences_df, Y_sequences_df, test_size=0.2, shuffle=False,random_state=20)

    model = Sequential()
    model.add(LSTM(50, input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error') 
    model.fit(X_train["input"], Y_train, epochs=20, batch_size=32) 
    initial_sequence = dfk[k].values[-sequence_length:] 
    predicted_data = list(initial_sequence)
    num_steps_to_predict = 10

    for step in range(num_steps_to_predict):
        sequence_length=len(initial_sequence)
        input_data = np.array(initial_sequence).reshape(1,sequence_length, 1)
        next_data_point = model.predict(input_data)
        predicted_data.append(next_data_point[0, 0])
        initial_sequence = initial_sequence[1:] + [next_data_point[0, 0]]
    predicted_data = predicted_data[:-num_steps_to_predict]
    last10 = predicted_data[-num_steps_to_predict:]
    pred2[k]=last10
X = dfk[["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"]]
X = pd.DataFrame(X,columns=["temperature","ph","salinity","dis_oxy","light","water_speed","nutrient","nutrient2"])
Y = dfk["size"]
train_X, test_X, train_y, test_y = train_test_split(X, Y, 
                                                    test_size = 0.25, 
                                                    random_state = 123)
model = LinearRegression()
model.fit(train_X, train_y)
pred_y = model.predict(pred2)
pred2["size"]=pred_y
pred2["date"]=dates
pred2["cluster"]=np.repeat(1,10)
print(pred2)
finals = pd.concat([pred2,finals])

print(finals)


# In[ ]:


finals["type"]="prediction"
df2["type"]="original"
combineS2 = pd.concat([finals,df2])
print(combineS2.head(2))


# In[ ]:


# Import back to InfluxDB

token = "bIq_bcybf3PNahATbM75_rx2-koAqxmMypxSjVbmy80NXzK5eXaeXGYRttGmzdN1KdsEtdR0c-QLPNGHAJ9c-Q=="
org = "None"
host = "https://us-east-1-1.aws.cloud2.influxdata.com"
database ="Maldives OH 2023"
client = InfluxDBClient3(host=host, token=token, org=org)

##### USE THIS TO DELETE DATA IF ACCIDENTALLY INPUT TWICE
# client.delete_api().delete(
#     predicate =' _measurement == "Test"',
#     start=datetime(1970,1,1),
#     stop=datetime(2100,1,1),
#     bucket=database,
#     org=org
# )

##### ONLY COULD RUN ONE TIME TO INITIATE THE DATASET IMPORT
# combineS2 = combineS2.to_json(orient="records")
# combineS2j = json.loads(combineS2j)
# for key in combineS2j:
#     point = Point("Test")
#     point.field("date",key["date"])
#     point.tag("latitude", key["latitude"])
#     point.tag("longitude",key["longtitude"])
#     point.field("temperature", key["temperature"])
#     point.field("ph", key["ph"])
#     point.field("salinity",key["salinity"])
#     point.field("dis_oxy",key["dis_oxy"])
#     point.field("light",key["light"])
#     point.field("water_speed",key["water_speed"])
#     point.field("nutrient", key["nutrient"])
#     point.field("nutrient2",key["nutrient2"])
#     point.field("occurrence", key["occurrence"])
#     point.field("type",key["type"])
#     point.filed("size",key["size"])
#     client.write(database=database, record=point)
# print("complete")

