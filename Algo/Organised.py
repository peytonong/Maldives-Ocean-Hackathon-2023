Class

import os, time
from influxdb_client_3 import InfluxDBClient3, Point
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report, roc_curve,roc_auc_score
from sklearn.datasets import load_digits
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd










#----------------------------------------------------------------------

#Input data to Influx using Jupyter

import os, time
from influxdb_client_3 import InfluxDBClient3, Point

token = "bIq_bcybf3PNahATbM75_rx2-koAqxmMypxSjVbmy80NXzK5eXaeXGYRttGmzdN1KdsEtdR0c-QLPNGHAJ9c-Q=="  ##token name delete first few useless word
org = "None"
host = "https://us-east-1-1.aws.cloud2.influxdata.com"  ##influx url until .com

client = InfluxDBClient3(host=host, token=token, org=org)

database="Maldives Ocean Hackathon" ##database name

data = {
  "point200000": {
    "location": "LOL",
    "species": "bees",
    "count": 28,
  }
}

for key in data:
  point = (
    Point("census")  ##table name
    .tag("location", data[key]["location"]) ##can two tag sets another (.tag(bla,data[key][“bla”])
    .field(data[key]["species"], data[key]["count"])
  )
  client.write(database=database, record=point)
  time.sleep(1) # separate points by 1 second

print("Complete.")

query = """SELECT *
FROM 'census'
WHERE time >= now() - interval '24 hours'
AND ('bees' IS NOT NULL OR 'ants' IS NOT NULL)"""

# Execute the query
table = client.query(query=query, database="Maldives Ocean Hackathon", language='sql') 

# Convert to dataframe
df = table.to_pandas().sort_values(by="time")
print(df)


#----------------------------------------------------------------------

#Data Cleaning

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df1 = pd.read_csv('Master_Dataset_LCK_2014-2020v2.csv')
df1["date"] = pd.to_datetime(df1["date"],format="%Y-%m-%d")
df1 = df1.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df1 = df1.dropna(subset=["date","temperature","ph","salinity","dis_oxy"])
df1 = df1.drop_duplicates()

Q1 = df1['temperature'].quantile(0.25)
Q3 = df1['temperature'].quantile(0.75)
IQR = Q3 - Q1
tp_lower_bound = Q1 - 2.0 * IQR
tp_upper_bound = Q3 + 2.0 * IQR
df1 = df1[(df1['temperature'] >= tp_lower_bound) & (df1['temperature'] <= tp_upper_bound)]
df1["Temperature Category"] = ""
df1.loc[df1["temperature"] > 23, "Temperature Category"] = "Hot"
df1.loc[(df1["temperature"] >= 19) & (df1["temperature"] <= 23), "Temperature Category"] = "Warm"
df1.loc[df1["temperature"] < 19, "Temperature Category"] = "Cold"

#create dummy data
df1["red algae"]="No"
df1.loc[(df1["salinity"] >28)&(df1["salinity"]<32)&(df1["temperature"]>18)&(df1["temperature"]<20) &(df1["ph"]>7)&(df1["ph"]<8), "red algae"] = "Yes"
print(len(df1.loc[df1["red algae"]=="Yes",]))

Salinity: 3%
Nutrient level: 200 μM NaNO3 and 20 μM NaH2PO4 (final concentration)
Light availability: (100 μmol photos m−2 s−1), operated on a 12 h on-and-off photoperiod
Water movement: aeration and culture seawater was updated every 2 days
Temperature: 19+-1 degree
ph: 7.0-7.5 (7.5 peak)
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5011508/ 
https://bioone.org/journals/arctic-antarctic-and-alpine-research/volume-39/issue-1/1523-0430(2007)39%5B65%3ATOPOTG%5D2.0.CO%3B2/The-Optimum-pH-of-the-Green-Snow-Algae-Chloromonas-Tughillensis/10.1657/1523-0430(2007)39[65:TOPOTG]2.0.CO;2.full 


#----------------------------------------------------------------------

#Data Grouping

import numpy as np
from sklearn.cluster import KMeans

# Assuming you have a DataFrame named 'df' with 'latitude' and 'longitude' columns
# Feature engineering: convert latitude and longitude to radians
df1['lat_rad'] = np.radians(df1['latitude'])
df1['lon_rad'] = np.radians(df1['longitude'])

# Combine latitude and longitude into a feature matrix
X = df1[['lat_rad', 'lon_rad']]

# Choose the number of clusters (K)
k = 4  # You can adjust this based on your data and needs

# Apply K-Means clustering
kmeans = KMeans(n_clusters=k,n_init=10)
df1['cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
import matplotlib.pyplot as plt

# Plot the data points with different colors for each cluster
for cluster_id in range(k):
    plt.scatter(df1[df1['cluster'] == cluster_id]['longitude'], df1[df1['cluster'] == cluster_id]['latitude'], label=f'Cluster {cluster_id}')

plt.legend()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


#----------------------------------------------------------------------

#Train and find best model

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report, roc_curve,roc_auc_score
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Setting SEED for reproducibility
SEED = 23
dfp=df1
accuracys_gbc=[]
accuracys_dt=[]
accuracys_rf=[]
accuracys_lr=[]
gbcm=[]
dtm=[]
rfm=[]
lrm=[]

for k in range(len(df1["cluster"].unique())):
    # Importing the dataset 
    dfk=""
    dfk = df1.loc[df1["cluster"]==k,]
    X = dfk[["ph","dis_oxy","salinity"]]
    X = pd.DataFrame(X,columns=["ph","dis_oxy","salinity"])
    Y = dfk["Temperature Category"]

    # Splitting dataset
    trainX, test_X, trainY, test_y = train_test_split(X, Y, 
                                                        test_size = 0.25, 
                                                        random_state = 123)
    ##
    
    gbc = GradientBoostingClassifier(n_estimators=1000,
                                     learning_rate=0.05,
                                     random_state=100,
                                     max_features=5 )
    gbcm.append(gbc)
    gbc.fit(trainX, trainY)
    pred_y = gbc.predict(test_X)
    acc = accuracy_score(test_y, pred_y)
    accuracys_gbc.append(acc)
    
    d=DecisionTreeClassifier()
    d.fit(trainX,trainY)
    dtm.append(d)
    pred_y=d.predict(test_X)
    acc = accuracy_score(test_y, pred_y)
    accuracys_dt.append(acc)
    
    rf=RandomForestClassifier()
    rf.fit(trainX,trainY)
    rfm.append(rf)
    pred_y=rf.predict(test_X)
    acc = accuracy_score(test_y, pred_y)
    accuracys_rf.append(acc)
    
    lr=LogisticRegression(solver='lbfgs',max_iter=1000)
    lr.fit(trainX,trainY)
    lrm.append(lr)
    pred_y=lr.predict(test_X)
    acc = accuracy_score(test_y, pred_y)
accuracys_lr.append(acc)

# svc = SVC(kernel="poly",degree=3,coef0=1,C=1.0) #rbf poly 
    
print(accuracys_gbc,"\n average: ",st.mean(accuracys_gbc),"\n")
print(accuracys_dt,"\n average: ",st.mean(accuracys_dt),"\n")
print(accuracys_rf,"\n average: ",st.mean(accuracys_rf),"\n")
print(accuracys_lr,"\n average: ",st.mean(accuracys_lr),"\n")


 #----------------------------------------------------------------------

#LSTM for each factors

## lstm for ph
df1.sort_values(by="date")

## create sequence used for lstm
# df1.set_index('date', inplace=True)
sequence_length = 10
input_sequences = []
target_sequences = []

for i in range(len(df1) - sequence_length):
    X_seq = df1["ph"].values[i]
    Y_seq = df1["ph"].values[i+sequence_length]
    X_dates = df1["date"].values[i]
    input_sequences.append({"dates":X_dates,"ph":X_seq})
    target_sequences.append(df1.iloc[i+sequence_length,9])
    
# Convert the lists to DataFrames if needed
X_sequences_df = pd.DataFrame(input_sequences)
Y_sequences_df = pd.Series(target_sequences, name='ph_target')

# Now you have X_sequences_df containing sequences of 'ph' values with dates and y_sequences_df with target values.
TrainX, TestX, TrainY, TestY = train_test_split(
    X_sequences_df, Y_sequences_df, test_size=0.2, shuffle=False,random_state=42)


##LSTM
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Initialize the model
model = Sequential()

# Add an LSTM layer with the input shape matching your sequence length
model.add(LSTM(50, input_shape=(sequence_length, 1)))

# Add a Dense output layer with one unit for regression
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')  # Use appropriate loss function

model.fit(TrainX['ph'], TrainY, epochs=20, batch_size=32)  # Adjust the number of epochs and batch size
## batch_size -> 32,64,128

# Evaluate the model on the test data
loss = model.evaluate(TestX['ph'], TestY)
print(f'Mean Squared Error on Test Data: {loss}')


# Initialize an empty sequence to start the prediction
initial_sequence = df1["ph"].values[-sequence_length:]  # Use the last sequence_length data points from your historical data

# Initialize a list to store the predicted data
predicted_data = list(initial_sequence)

# Number of future time steps to predict (e.g., 10 years with monthly data)
num_steps_to_predict = 100 # Adjust as needed

for step in range(num_steps_to_predict):
    sequence_length=len(initial_sequence)
    # Reshape the initial_sequence for the model's input shape
    #reshape (batch_size,sequence_length,numberoffeature)
    input_data = np.array(initial_sequence).reshape(1, sequence_length, 1)
    
    # Predict the next data point
    next_data_point = model.predict(input_data)
    
    # Append the prediction to the predicted_data
    predicted_data.append(next_data_point[0, 0])  # Assuming a single-value output
    
    # Update the initial_sequence for the next prediction
    initial_sequence = initial_sequence[1:] + [next_data_point[0, 0]]

print(len(initial_sequence))
print(sequence_length)
print("predict how much: ",len(predicted_data)-len(initial_sequence))
## new 24 datas (but only step for 12 so last 12 is wrong one)
## need to remove last -number_of_steps
predicted_data = predicted_data[:-num_steps_to_predict]
print("predict how much?: ",len(predicted_data)-len(initial_sequence))
print("last ",num_steps_to_predict,": ",predicted_data[-num_steps_to_predict:])
# The predicted_data list now contains the predictions for the next 10 years

