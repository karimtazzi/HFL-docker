import flwr as fl
import os
import pickle
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, NDArrays, Scalar
from typing import Dict, List, Tuple
from functools import reduce
import time

# Retrieve environment variables
server_path = os.environ.get("Server_path")
BS_ID = int(os.environ.get("BS_ID", 3))

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
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
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
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
