import flwr as fl
import os
import pickle
import numpy as np
from functools import reduce
from typing import List, Tuple

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays

from typing import Dict, List, Tuple
from flwr.common import Scalar

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
    """Load all weights from the specified directory."""
    weights_list = []  # List to store weights
    
    # Iterate over all files in the directory
    for filename in os.listdir(server_path):
        file_path = os.path.join(server_path, filename)
        
        # Ensure the file is a weights file (could add specific filtering if needed)
        if os.path.isfile(file_path) and filename.endswith(".obj"):  # Assuming weights files end with .obj
            # Load the weights from the file
            with open(file_path, 'rb') as h:
                weights = pickle.load(h)
                weights_list.append((parameters_to_ndarrays(weights[0]), weights[1]))  # Add weights to the list
            
            # Delete the file after loading
            os.remove(file_path)
            print(f"Deleted weights file: {filename}")
    
    return weights_list

# Load the weights results
weights_results = load_all_weights(Server_path)

def Aggregation(weights_results: List[Tuple[NDArrays, int]]):
    """Aggregate the weights results."""
    aggregated_ndarrays = aggregate(weights_results)
    parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
    return parameters_aggregated
metrics_aggregated = {}
fit_metrics = [(weights[2], weights[1]) for weights in weights_results]
#metrics_aggregated = fit_metrics_aggregation_fn(fit_metrics)

# Perform the aggregation
aggregated_weights = Aggregation(weights_results)

if not os.path.exists(S_weights_path):
    os.makedirs(S_weights_path)
# Save the aggregated weights
with open(os.path.join(S_weights_path, "Global_weights.obj"), 'wb') as h:
    pickle.dump(aggregated_weights, h)

print(f"Aggregated weights saved to {os.path.join(S_weights_path, 'Global_weights.obj')}")
