from sympy import Integer
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
import hashlib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

#url = 'https://raw.githubusercontent.com/wiktorDeka/WykrywanieAnomalii/main/reduced_dataset_sdn.csv'
url = 'https://raw.githubusercontent.com/wiktorDeka/WykrywanieAnomalii/main/dataset_sdn.csv'
s = requests.get(url).content

counters = pd.read_csv(io.StringIO(s.decode('utf-8')))
counters.dt = pd.to_datetime(counters.dt, unit='s')
counters.set_index('dt', inplace=True)
counters.fillna(value=0, inplace=True)

#ip embedding



#target encoding
#bucketing
src_label_encoder = LabelEncoder()
counters['src_ip_encoded'] = src_label_encoder.fit_transform(counters['src'])

dst_label_encoder = LabelEncoder()
counters['dst_ip_encoded'] = dst_label_encoder.fit_transform(counters['dst'])

label_encoder = LabelEncoder()
counters['Protocol'] = label_encoder.fit_transform(counters['Protocol'])

counters.drop('src', axis=1, inplace=True)
counters.drop('dst', axis=1, inplace=True)
counters
#print(counters.columns)
#hash semantyczny

#isolation forest / SVM
#pca

# Specify the percentage of data you want to use for training
counters_withoutLabel = counters.drop('label',axis=1)


percentage = 0.2
X_train, X_test, y_train, y_test = train_test_split(counters_withoutLabel, counters['label'], train_size=percentage, random_state=42)
# Split data into training features and target variable

# Isolation forest
las_izol = IsolationForest()
outliers_pred = las_izol.fit_predict(X_train)

X_train['wspolczynnik_anomalnosci'] = las_izol.decision_function(X_train)
X_train['rozpoznanie'] = outliers_pred

top_features = ['pktperflow', 'bytecount', 'packetins']

""" # Plotting the 3D scatter plot
fig = px.scatter_3d(X_train, x=top_features[0], y=top_features[1], z=top_features[2], color='rozpoznanie',
                    title='Wynik działania lasu izolującego (Top 3 Features)')
fig.show() """

counters_withoutLabel = counters.drop('label',axis=1)

print("Dokładność na zbiorze uczącym:", (y_train != -1).mean())
anomaly_predictions = las_izol.predict(counters_withoutLabel)
# Calculate accuracy
accuracy = (anomaly_predictions == -1).mean()
print(f"Dokładność na zbiorze całym: {accuracy}")