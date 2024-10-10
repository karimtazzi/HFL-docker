import flwr as fl
import os
import pickle
import numpy as np
from functools import reduce
from typing import List, Tuple

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays

from typing import Dict, List, Tuple
from flwr.common import Scalar



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



def fit_metrics_aggregation_fn(
    fit_metrics: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    """Aggregate fit metrics using weighted averages based on client data.

    Parameters
    ----------
    fit_metrics : List[Tuple[int, Dict[str, Scalar]]]
        A list where each element is a tuple containing the number of examples
        used for training on a client and a dictionary with the client’s metrics.

    Returns
    -------
    Dict[str, Scalar]
        Aggregated metrics (e.g., weighted average accuracy, loss).
    """
    # Initialize variables for weighted sum of metrics
    total_examples = 0
    weighted_loss_sum = 0.0
    weighted_accuracy_sum = 0.0

    # Iterate over client metrics and accumulate weighted sums
    for num_examples, metrics in fit_metrics:
        total_examples += num_examples
        weighted_loss_sum += metrics["loss"] * num_examples
        weighted_accuracy_sum += metrics["accuracy"] * num_examples

    # Compute weighted averages
    avg_loss = weighted_loss_sum / total_examples if total_examples > 0 else 0.0
    avg_accuracy = weighted_accuracy_sum / total_examples if total_examples > 0 else 0.0

    # Return the aggregated metrics
    return {
        "loss": avg_loss,
        "accuracy": avg_accuracy,
    }

# Fetch paths from environment variables
Server_path = os.environ.get("Server_path")
S_weights_path = os.environ.get("Gserver_path")

def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

def load_all_weights(server_path: str) -> List[Tuple[NDArrays, int]]:
    """Load all weights from the specified directory and delete files after reading."""
    weights_list = []  # List to store weights
    
    # Iterate over all files in the directory
    for filename in os.listdir(server_path):
        file_path = os.path.join(server_path, filename)
        
        # Ensure the file is a weights file (assuming weights files end with .obj)
        if os.path.isfile(file_path) and filename.endswith(".obj"):
            try:
                # Load the weights from the file
                with open(file_path, 'rb') as h:
                    weights = pickle.load(h)
                    weights_list.append([weights[0], weights[1], weights[2]])  # Add weights to the list
                
                # Delete the file after loading
                os.remove(file_path)
                print(f"Deleted weights file: {filename}")
            
            except (OSError, IOError) as e:
                print(f"Error loading or deleting file {filename}: {e}")
    
    return weights_list
from flask import Flask, request, jsonify

app = Flask(__name__)
def receive_data():
    data = request.get_json()  # Get the JSON data sent from Container A
    print(f"Received data: {data["rnd"]}")
    return data["rnd"]
print("yesssssssssssssssssssss")
print(receive_data())
# Load the weights results

weights_results = load_all_weights(Server_path)
def Aggregation(weights_results: List[Tuple[NDArrays, int]]):
    """Aggregate the weights results."""
    aggregated_ndarrays = aggregate(weights_results)
    parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
    return parameters_aggregated

weights_res=[(weights[0], weights[1]) for weights in weights_results]
rnd=weights_results[0][2]
# Perform the aggregation
aggregated_weights = Aggregation(weights_res)
res=[aggregated_weights, {}]
if not os.path.exists(S_weights_path):
    os.makedirs(S_weights_path)
# Save the aggregated weights
with open(os.path.join(S_weights_path, "Global_weights_"+str(rnd)+".obj"), 'wb') as h:
    pickle.dump(res, h)

print(f"Aggregated weights saved to {os.path.join(S_weights_path,  "Global_weights_"+str(rnd)+".obj")}")
