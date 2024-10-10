import flwr as fl
import sys
import numpy as np
import os
import pickle
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy.aggregate import aggregate
from logging import log, WARNING
from typing import Dict, List, Tuple
from flwr.common import Scalar


# Retrieve environment variables
Server_path = os.environ.get("Server_path")
BS_ID = int(os.environ.get("BS_ID", 3))
gserver_path = os.environ.get("Gserver_path")


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


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        # Store model weights and metrics
        weights_metrices = []

        # Convert results to numpy arrays and aggregate them
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Save aggregated weights in metrics list
        weights_metrices.append(aggregated_ndarrays)

        # Example usage correction (fixed variable `some`, which was undefined)
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        weights_metrices.append(total_examples)
        weights_metrices.append(rnd)
        # Call the base class method to perform the aggregation
        #aggregated_result = super().aggregate_fit(rnd, results, failures)
        # Save the aggregated weights locally to the server path
        S_weights_path = os.path.join(Server_path)
        if not os.path.exists(S_weights_path):
            os.makedirs(S_weights_path)

        with open(os.path.join(S_weights_path, f"BS_{BS_ID}_weights_round_{rnd}.obj"), 'wb') as h:
            pickle.dump(weights_metrices, h)

        # Load global weights after the first round
        if rnd >= 1:
            global_weights_path = os.path.join(gserver_path, "Global_weights.obj")
            with open(global_weights_path, 'rb') as h:
                aggregated_result = pickle.load(h), {}
        else:
            aggregated_result = super().aggregate_fit(rnd, results, failures)

        aggregated_weights, aggregated_metrics = aggregated_result


        return aggregated_weights, aggregated_metrics

# Create strategy and run the Flower server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = 'intermediate_server2:5004', 
        config=fl.server.ServerConfig(num_rounds=3) ,
        strategy = strategy
)