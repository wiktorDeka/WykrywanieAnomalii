from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.decomposition import PCA
import pandas as pd


def extract_last_byte(ip):
    try:
        return int(ip.split('.')[-1])
    except Exception as e:
        print(f'Error parsing IP {ip} : {e}')
        return None


def encode_data(network_data, ip_encoding, variance_threshold=None, drop=False):
    if drop:
        columns_to_stay = ['bytecount', 'packetins', 'pktperflow', 'byteperflow', 'Protocol', 'label']
        network_data = network_data[columns_to_stay]
    if ip_encoding == 0:
        # label_encoding
        label_encoder = LabelEncoder()
        if not drop:
            network_data['src_ip_encoded'] = label_encoder.fit_transform(network_data['src'])
            network_data.drop('src', axis=1, inplace=True)

            network_data['dst_ip_encoded'] = label_encoder.fit_transform(network_data['dst'])
            network_data.drop('dst', axis=1, inplace=True)

        network_data['protocol_encoded'] = label_encoder.fit_transform(network_data['Protocol'])
        network_data.drop('Protocol', axis=1, inplace=True)

        network_data.dropna(inplace=True)

        y = network_data['label']
        X = network_data.drop('label', axis=1)

    elif ip_encoding == 1:
        # target encoding
        te = TargetEncoder(smooth='auto', target_type='binary')
        y = network_data['label']
        X = network_data.drop('label', axis=1)

        if not drop:
            target = network_data['src']
        else:
            target = network_data['Protocol']

        X = te.fit_transform(X, target)

    elif ip_encoding == 2:
        # encode as last byte of ip addr
        if not drop:
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

    elif ip_encoding == 3:
        # One-Hot Encoding
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        if not drop:
            src_encoded = one_hot_encoder.fit_transform(network_data[['src']])
            dst_encoded = one_hot_encoder.fit_transform(network_data[['dst']])
            src_columns = [f'src_{i}' for i in range(src_encoded.shape[1])]
            dst_columns = [f'dst_{i}' for i in range(dst_encoded.shape[1])]
            src_df = pd.DataFrame(src_encoded, columns=src_columns)
            dst_df = pd.DataFrame(dst_encoded, columns=dst_columns)
            network_data = pd.concat([network_data, src_df, dst_df], axis=1)
            network_data.drop(['src', 'dst'], axis=1, inplace=True)

        protocol_encoded = one_hot_encoder.fit_transform(network_data[['Protocol']])
        protocol_columns = [f'protocol_{i}' for i in range(protocol_encoded.shape[1])]
        protocol_df = pd.DataFrame(protocol_encoded, columns=protocol_columns)

        network_data = pd.concat([network_data, protocol_df], axis=1)
        network_data.drop(['Protocol'], axis=1, inplace=True)
        network_data.dropna(inplace=True)

        y = network_data['label']
        X = network_data.drop('label', axis=1)

    else:
        print('invalid encoding')
        exit(-1)

    if variance_threshold is not None:
        pca = PCA(n_components=variance_threshold)
        X = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    pass
