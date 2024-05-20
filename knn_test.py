from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
import pandas as pd
import numpy as np


def extract_last_byte(ip):
    try:
        return int(ip.split('.')[-1])
    except Exception as e:
        print(f'Error parsing IP {ip} : {e}')
        return None


def encode_data(network_data, ip_encoding):
    if ip_encoding == 0:
        # label_encoding
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
        # target encoding
        te = TargetEncoder(smooth='auto', target_type='binary')
        y = network_data['label']
        X = network_data.drop('label', axis=1)
        target = network_data['src']

        X_trans = te.fit_transform(X, target)
        X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.3, random_state=0)

    elif ip_encoding == 2:
        # encode as last byte of ip addr
        network_data['src_ip_encoded'] = network_data['src'].apply(extract_last_byte)
        network_data.drop('src', axis=1, inplace=True)
        network_data['dst_ip_encoded'] = network_data['dst'].apply(extract_last_byte)
        network_data.drop('dst', axis=1, inplace=True)

        label_encoder = LabelEncoder()
        network_data['protocol_encoded'] = label_encoder.fit_transform(network_data['Protocol'])
        network_data.drop('Protocol', axis=1, inplace=True)

        network_data.dropna(inplace=True)

        y = network_data['label']
        X = network_data.drop('label', axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    elif ip_encoding == 3:
        # One-Hot Encoding
        one_hot_encoder = OneHotEncoder(sparse_output=False)

        src_encoded = one_hot_encoder.fit_transform(network_data[['src']])
        dst_encoded = one_hot_encoder.fit_transform(network_data[['dst']])
        protocol_encoded = one_hot_encoder.fit_transform(network_data[['Protocol']])

        src_columns = [f'src_{i}' for i in range(src_encoded.shape[1])]
        dst_columns = [f'dst_{i}' for i in range(dst_encoded.shape[1])]
        protocol_columns = [f'protocol_{i}' for i in range(protocol_encoded.shape[1])]

        src_df = pd.DataFrame(src_encoded, columns=src_columns)
        dst_df = pd.DataFrame(dst_encoded, columns=dst_columns)
        protocol_df = pd.DataFrame(protocol_encoded, columns=protocol_columns)

        network_data = pd.concat([network_data, src_df, dst_df, protocol_df], axis=1)
        network_data.drop(['src', 'dst', 'Protocol'], axis=1, inplace=True)
        network_data.dropna(inplace=True)

        y = network_data['label']
        X = network_data.drop('label', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    else:
        print('invalid encoding')
        exit(-1)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    network_data = pd.read_csv('dataset_sdn.csv')
    # network_data = pd.read_csv('reduced_dataset_sdn.csv')

    # replace or delete all NaN values from dataset
    # print(np.count_nonzero(network_data.isnull().values))
    null_info = network_data.isna().sum()
    print(null_info)

    ip_encoding = 3
    X_train, X_test, y_train, y_test = encode_data(network_data, ip_encoding)

    K = []
    training = []
    test = []
    scores = {}

    for k in range(2, 21):
        clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)

        test_score = clf.score(X_test, y_test)
        print(k, '', test_score)
