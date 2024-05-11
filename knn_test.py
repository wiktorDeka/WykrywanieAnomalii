from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

network_data = pd.read_csv('reduced_dataset_sdn.csv')

src_label_encoder = LabelEncoder()
network_data['src_ip_encoded'] = src_label_encoder.fit_transform(network_data['src'])
network_data.drop('src', axis=1, inplace=True)

dst_label_encoder = LabelEncoder()
network_data['dst_ip_encoded'] = dst_label_encoder.fit_transform(network_data['dst'])
network_data.drop('dst', axis=1, inplace=True)


network_data['protocol_encoded'] = dst_label_encoder.fit_transform(network_data['Protocol'])
network_data.drop('Protocol', axis=1, inplace=True)
y = network_data['label']
X = network_data.drop('label', axis=1)
print(y)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

K = []
training = []
test = []
scores = {}

for k in range(2, 21):
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)

    test_score = clf.score(X_test, y_test)
    print(k, '', test_score)
