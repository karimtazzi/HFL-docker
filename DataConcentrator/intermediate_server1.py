import flwr as fl
import os
import pickle
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays, Scalar
from typing import Dict, List, Tuple
from functools import reduce
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Input, Dropout,
    Bidirectional, Conv1D, MaxPooling1D,
    LSTM
)
import time

# Retrieve environment variables
server_path = os.environ.get("Server_path")
BS_ID = int(os.environ.get("BS_ID", 3))

def create_cnn_model(X_train_shape=70, n_classes=2):
    kernel_size = 3
    filters1 = 64
    filters2 = 128
    pool_size = 2

    cnn_model = Sequential([

        Input((X_train_shape, 1)),

        Conv1D(filters1, kernel_size, padding='same', activation='relu', strides=1),
        Conv1D(filters1, kernel_size, padding='same', activation='relu', strides=1),
        MaxPooling1D(pool_size=pool_size),
        Dropout(0.5),

        Conv1D(filters2, kernel_size, padding='same', activation='relu', strides=1),
        Conv1D(filters2, kernel_size, padding='same', activation='relu', strides=1),
        MaxPooling1D(pool_size=pool_size),

        Bidirectional(LSTM(32)),

        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(n_classes, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return cnn_model

import pandas as pd

model = create_cnn_model()
server_PATH = os.environ.get("Server_path")
if server_PATH is None:
    raise ValueError("Server_path environment variable not set.")    
server_data_path = os.path.join(server_PATH, "server.csv")
DATA = pd.read_csv(server_data_path)

def detect_malicious_clients(model, ll, data):
    new_ll = []  # Initialize the new list to store valid weights
    X_ev = data.drop(['Label'], axis=1).values  # Features
    y_ev = data['Label'].values  # Labels

    X_ev = X_ev.reshape(X_ev.shape[0], X_ev.shape[1], 1)  # Reshape for CNN input

    # Iterate over the list of weights
    for w in ll:
        model.set_weights(w[0])  # Set the model weights
        loss, acc = model.evaluate(X_ev, y_ev)  # Evaluate the model

        if loss <= 1:  # Check if the loss is acceptable
            new_ll.append(w)  # Add the valid weight to the list

    return new_ll  # Return the list of valid weights


# Renamed to avoid conflict with flwr's aggregate import
def aggregate_weights(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average of NDArrays."""
    num_examples_total = sum(num_examples for _, num_examples in results)

    # Create weighted layers
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

# Load all weights from files
def load_all_weights(server_path: str, rnd: int) -> List[Tuple[NDArrays, int]]:
    """Load all weights from the specified directory."""
    weights_list = []

    for filename in os.listdir(server_path):
        file_path = os.path.join(server_path, filename)
        if os.path.isfile(file_path) and filename.endswith(f"round_{rnd}.obj"):
            try:
                with open(file_path, 'rb') as h:
                    weights = pickle.load(h)
                    weights_list.append((weights[0], weights[1]))  # Assuming (weights, num_examples)
            except (OSError, IOError) as e:
                print(f"Error loading file {filename}: {e}")

    return weights_list

# Aggregation function for weights
def aggregate_results(weights_results: List[Tuple[NDArrays, int]]):
    """Aggregate the weights results."""
    aggregated_ndarrays = aggregate_weights(weights_results)
    parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
    return parameters_aggregated

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results,
        failures
    ) -> Tuple[fl.common.Parameters, Dict[str, Scalar]]:
        """Aggregate model weights and save them locally."""

        # Convert client results to numpy arrays and aggregate them
        old_weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Detect and remove malicious clients
        print("############################ Start Malicious clients detection for Base station 1 ####################")
        weights_results = detect_malicious_clients(model, old_weights_results, DATA)
        
        # Aggregate weights
        print("############################ Start aggregation process for Base station 1 ############################")
        aggregated_ndarrays = aggregate_weights(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Prepare to save weights and metrics
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        weights_metrics = [aggregated_ndarrays, total_examples, rnd]

        # Ensure directory exists for saving weights
        s_weights_path = os.path.join(server_path)
        os.makedirs(s_weights_path, exist_ok=True)

        # Save weights for the current round
        with open(os.path.join(s_weights_path, f"BS_{BS_ID}_weights_round_{rnd}.obj"), 'wb') as h:
            pickle.dump(weights_metrics, h)
        print("############################ parameters of first aggregation of BS1 are stored in {s_weights_path} ###############")
        # Simulate waiting
        time.sleep(25)

        # Load and aggregate all weights from the directory
        weights_results = load_all_weights(s_weights_path, rnd)
        aggregated_weights = aggregate_results(weights_results)
        aggregated_metrics = {}

        # Save the aggregated weights
        with open(os.path.join(s_weights_path, f"Global_weights_{rnd}.obj"), 'wb') as h:
            pickle.dump(aggregated_weights, h)

        return aggregated_weights, aggregated_metrics

# Create strategy and run the Flower server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
    server_address='intermediate_server1:5003',
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy
)
