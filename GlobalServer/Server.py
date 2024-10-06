import flwr as fl
import os
import pickle
import numpy as np
from functools import reduce
from typing import List, Tuple

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays

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
                weights_list.append((parameters_to_ndarrays(weights[0]), 500))  # Add weights to the list
            
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

# Perform the aggregation
aggregated_weights = Aggregation(weights_results)

if not os.path.exists(S_weights_path):
    os.makedirs(S_weights_path)
# Save the aggregated weights
with open(os.path.join(S_weights_path, "Global_weights.obj"), 'wb') as h:
    pickle.dump(aggregated_weights, h)

print(f"Aggregated weights saved to {os.path.join(S_weights_path, 'Global_weights.obj')}")
