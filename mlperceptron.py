import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from data_encoder import encode_data


def evaluate_mlp(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', learning_rate_init=0.001):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, learning_rate_init=learning_rate_init, max_iter=500)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    print(f'Hidden layers: {hidden_layer_sizes}, Activation: {activation}, Solver: {solver}, Learning rate: {learning_rate_init}')
    print(f'Test accuracy: {test_score:.4f}')
    return test_score


network_data = pd.read_csv('dataset_sdn.csv')
network_data = network_data.dropna()
ip_encoding = 3  # one hot
X_train, X_test, y_train, y_test = encode_data(network_data, ip_encoding, drop=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

hyperparameters = [
    {'hidden_layer_sizes': (5, 5), 'activation': 'logistic', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (10, 10), 'activation': 'logistic', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (5, 5), 'activation': 'logistic', 'solver': 'adam', 'learning_rate_init': 0.0001},
    {'hidden_layer_sizes': (10, 10), 'activation': 'logistic', 'solver': 'adam', 'learning_rate_init': 0.0001},
    {'hidden_layer_sizes': (5, 5), 'activation': 'logistic', 'solver': 'sgd', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (10, 10), 'activation': 'logistic', 'solver': 'sgd', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (5, 5), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (10, 10), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (5, 5), 'activation': 'tanh', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (10, 10), 'activation': 'tanh', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (5, 5), 'activation': 'relu', 'solver': 'sgd', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (10, 10), 'activation': 'relu', 'solver': 'sgd', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (5, 5), 'activation': 'tanh', 'solver': 'sgd', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (10, 10), 'activation': 'tanh', 'solver': 'sgd', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (50, 50), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (100, 100), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (50, 50), 'activation': 'tanh', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (100, 100), 'activation': 'tanh', 'solver': 'adam', 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (50, 50), 'activation': 'relu', 'solver': 'sgd', 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (100, 100), 'activation': 'relu', 'solver': 'sgd', 'learning_rate_init': 0.01}
]

best_score = 0
best_params = {}
for params in hyperparameters:
    score = evaluate_mlp(**params)
    if score > best_score:
        best_score = score
        best_params = params

print(f'Best parameters: {best_params}')
print(f'Best accuracy: {best_score:.4f}')
