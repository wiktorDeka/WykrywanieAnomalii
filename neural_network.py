import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from data_encoder import encode_data


def perceptron_routine():
    data = pd.read_csv('dataset_sdn.csv')
    X_train, X_test, y_train, y_test = encode_data(data, 3)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    max_iter_range = range(500, 2000, 100)
    tol_values = [5e-4, 1e-3, 2e-3, 4e-3]
    results = []

    for max_iter in max_iter_range:
        for tol in tol_values:
            perceptron = Perceptron(max_iter=max_iter, tol=tol)
            perceptron.fit(X_train, y_train)
            y_pred = perceptron.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((max_iter, tol, accuracy))

    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)

    for max_iter, tol, accuracy in sorted_results:
        print(f"max_iter: {max_iter}, tol: {tol}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    perceptron_routine()
