import flwr as fl
import os
import pickle
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy.aggregate import aggregate
from typing import Dict, List, Tuple
from flwr.common import Scalar
from GlobalServer/Server import load_all_weights, aggregation
# Retrieve environment variables
Server_path = os.environ.get("Server_path")
BS_ID = int(os.environ.get("BS_ID", 3))
gserver_path = os.environ.get("Gserver_path")

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
        time.sleep(30)
        weights_results = load_all_weights(server_path, rnd)
        aggregated_weights = Aggregation(weights_results) 
        if not os.path.exists(s_weights_path):
            os.makedirs(s_weights_path)
        # Save the aggregated weights
        with open(os.path.join(s_weights_path, f"Global_weights_{rnd}.obj"), 'wb') as h:
            pickle.dump(res, h)
        return aggregated_weights, aggregated_metrics

# Create strategy and run the Flower server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
    server_address='intermediate_server1:5003',
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
