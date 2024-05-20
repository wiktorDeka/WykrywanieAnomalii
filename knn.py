from data_encoder import encode_data
import pandas as pd
from sklearn import neighbors


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
