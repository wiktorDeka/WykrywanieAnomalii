from data_encoder import encode_data
from sklearn import svm
import pandas as pd

if __name__ == '__main__':
    # model does not seem to work with not reducet dataset
    # network_data = pd.read_csv('dataset_sdn.csv')
    network_data = pd.read_csv('reduced_dataset_sdn.csv')

    # replace or delete all NaN values from dataset
    # print(np.count_nonzero(network_data.isnull().values))
    null_info = network_data.isna().sum()
    print(null_info)

    # target_encoding is great (1)
    ip_encoding = 1
    X_train, X_test, y_train, y_test = encode_data(network_data, ip_encoding)
    for i in range(1, 20, 1):
        actual_gamma = i / 100
        ksvm = svm.SVC(kernel='rbf', gamma=actual_gamma)
        # Train the model on the training data
        ksvm.fit(X_train, y_train)
        # Evaluate the model on the test data
        accuracy = ksvm.score(X_test, y_test)
        print(f'Accuracy with gamma {actual_gamma} :', accuracy)
    for j in range(1, 20, 1):
        for i in range(1, 20, 1):
            actual_gamma = i / 100
            actual_nu = j / 100
            clf = svm.OneClassSVM(kernel='rbf', gamma=actual_gamma, nu=actual_nu)
            clf.fit(X_train)

            y_pred_test = clf.predict(X_test)
            y_pred_test = [0 if y == 1 else 1 for y in y_pred_test]
            n_error_test = (y_pred_test != y_test).sum()
            test_accuracy = (len(y_test) - n_error_test) / len(y_test)
            print(f'One-Class SVM accuracy with gamma {actual_gamma} and nu {actual_nu}:', test_accuracy)
