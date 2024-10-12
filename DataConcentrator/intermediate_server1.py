import flwr as fl
import os
import pickle
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy.aggregate import aggregate
from typing import Dict, List, Tuple
from flwr.common import Scalar

# Retrieve environment variables
Server_path = os.environ.get("Server_path")
BS_ID = int(os.environ.get("BS_ID", 3))
gserver_path = os.environ.get("Gserver_path")

def signal_completion():
    with open('/data/gServer/completion.flag', 'w') as f:
        f.write('done')

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


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results,
        failures
    ) -> Tuple[fl.common.Parameters, Dict[str, Scalar]]:
        """Aggregate model weights and save them locally."""

        # Store aggregated weights and metrics
        weights_metrics = []

        # Convert client results to numpy arrays and aggregate them
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Save aggregated weights and metrics
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        weights_metrics.append(aggregated_ndarrays)
        weights_metrics.append(total_examples)
        weights_metrics.append(rnd)

        # Ensure directory exists for saving weights
        s_weights_path = os.path.join(Server_path)
        if not os.path.exists(s_weights_path):
            os.makedirs(s_weights_path)

        # Save weights for the current round
        with open(os.path.join(s_weights_path, f"BS_{BS_ID}_weights_round_{rnd}.obj"), 'wb') as h:
            pickle.dump(weights_metrics, h)

        # Save current round number
        with open(os.path.join(s_weights_path, "rnd.obj"), 'wb') as hh:
            pickle.dump(rnd, hh)

        # Load global weights from global server if they exist for the current round
        global_weights_path = os.path.join(gserver_path, f"Global_weights{rnd}.obj")
        try:
            with open(global_weights_path, 'rb') as h:
                aggregated_result = pickle.load(h), {}
            aggregated_weights, aggregated_metrics = aggregated_result
        except FileNotFoundError:
            print(f"Global weights for round {rnd} not found. Using locally aggregated weights.")
            aggregated_weights = parameters_aggregated
            aggregated_metrics = {}

        return aggregated_weights, aggregated_metrics


rnd_path = os.path.join(s_weights_path, "rnd.obj")
with open(rnd_path, 'rb') as h:
        rnd = pickle.load(h)
if rnd==3:
    signal_completion()

# Create strategy and run the Flower server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
    server_address='intermediate_server1:5003',
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
