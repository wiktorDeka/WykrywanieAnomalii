from data_encoder import encode_data
import pandas as pd
from sklearn import neighbors
import numpy as np
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
    network_data = pd.read_csv('dataset_sdn.csv')
    # network_data = pd.read_csv('reduced_dataset_sdn.csv')

    network_data.isna().sum()

    ip_encoding = 3
    X_train, X_test, y_train, y_test = encode_data(network_data, ip_encoding, drop=True)

    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)
    K = []
    training = []
    test = []
    scores = {}

    for k in range(2, 21):
        clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(clf, X, Y, cv=5)

        avg_cv_score = cv_scores.mean()
        print("Cross-Validation Score (k={}): {}".format(k, avg_cv_score))
