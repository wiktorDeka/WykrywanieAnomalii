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
    ksvm = svm.SVC(kernel='rbf', gamma=0.1)
    # Train the model on the training data
    ksvm.fit(X_train, y_train)
    # Evaluate the model on the test data
    accuracy = ksvm.score(X_test, y_test)
    print('Accuracy:', accuracy)
