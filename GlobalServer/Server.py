import flwr as fl
import os
import pickle
import numpy as np
from functools import reduce
from typing import List, Tuple, Dict
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays, Scalar
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Input, Dropout,
    Bidirectional, Conv1D, MaxPooling1D,
    LSTM
)
import time

# Model creation function
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

# Create model and save initial weights
model = create_cnn_model()
initial_weights = fl.common.ndarrays_to_parameters(model.get_weights())

# Ensure the server path exists
s_weights_path = os.environ.get("Gserver_path")
if s_weights_path:
    with open(os.path.join(s_weights_path, "Global_weights_0.obj"), 'wb') as file:
        pickle.dump(initial_weights, file)

# Function for aggregating fit metrics
def fit_metrics_aggregation_fn(
    fit_metrics: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """Aggregate fit metrics using weighted averages based on client data."""

    total_examples = 0
    weighted_loss_sum = 0.0
    weighted_accuracy_sum = 0.0

    # Accumulate weighted sums
    for num_examples, metrics in fit_metrics:
        total_examples += num_examples
        weighted_loss_sum += metrics["loss"] * num_examples
        weighted_accuracy_sum += metrics["accuracy"] * num_examples

    avg_loss = weighted_loss_sum / total_examples if total_examples > 0 else 0.0
    avg_accuracy = weighted_accuracy_sum / total_examples if total_examples > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": avg_accuracy,
    }

# Fetch paths from environment variables
server_path = os.environ.get("Server_path")
s_weights_path = os.environ.get("Gserver_path")

# Aggregate function for federated learning
def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average of NDArrays."""
    num_examples_total = sum(num_examples for (_, num_examples) in results)

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
        if os.path.isfile(file_path) and filename.endswith(f"{rnd}.obj"):
            try:
                with open(file_path, 'rb') as h:
                    weights = pickle.load(h)
                    weights_list.append((weights[0], weights[1]))  # Assuming (weights, num_examples)
            except (OSError, IOError) as e:
                print(f"Error loading file {filename}: {e}")

    return weights_list


# Aggregation function for weights
def Aggregation(weights_results: List[Tuple[NDArrays, int]]):
    """Aggregate the weights results."""
    aggregated_ndarrays = aggregate(weights_results)
    parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
    return parameters_aggregated


