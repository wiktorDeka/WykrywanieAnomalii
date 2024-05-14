from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
import pandas as pd
import numpy as np
# network_data = pd.read_csv('reduced_dataset_sdn.csv')

network_data = pd.read_csv('dataset_sdn.csv')
ip_encoding = 1

if ip_encoding == 0:
    label_encoder = LabelEncoder()
    network_data['src_ip_encoded'] = label_encoder.fit_transform(network_data['src'])
    network_data.drop('src', axis=1, inplace=True)

    network_data['dst_ip_encoded'] = label_encoder.fit_transform(network_data['dst'])
    network_data.drop('dst', axis=1, inplace=True)

    network_data['protocol_encoded'] = label_encoder.fit_transform(network_data['Protocol'])
    network_data.drop('Protocol', axis=1, inplace=True)

    network_data.dropna(inplace=True)

    y = network_data['label']
    X = network_data.drop('label', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


elif ip_encoding == 1:
    te = TargetEncoder(smooth='auto', target_type='binary')
    y = network_data['label']
    X = network_data.drop('label', axis=1)
    target = network_data['src']

    X_trans = te.fit_transform(X, target)
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.3, random_state=0)

# replace or delete all NaN values from dataset
# print(np.count_nonzero(network_data.isnull().values))
print(network_data.isna().sum())

K = []
training = []
test = []
scores = {}

for k in range(2, 21):
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)

    test_score = clf.score(X_test, y_test)
    print(k, '', test_score)
