import flwr as fl
import sys
import numpy as np
import os
import pickle
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy.aggregate import aggregate
from logging import log, WARNING

# Retrieve environment variables
Server_path = os.environ.get("Server_path")
BS_ID = int(os.environ.get("BS_ID", 3))
gserver_path = os.environ.get("Gserver_path")

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
        weights_metrices.append(parameters_aggregated )

        # Example usage correction (fixed variable `some`, which was undefined)
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        weights_metrices.append(total_examples)

        # Aggregate custom metrics if aggregation function was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif rnd == 1:  # Log this warning only in the first round
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        
        # Append aggregated metrics to the list
        weights_metrices.append(metrics_aggregated)


        # Call the base class method to perform the aggregation
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        aggregated_weights, aggregated_metrics = aggregated_result

        # Save the aggregated weights locally to the server path
        S_weights_path = os.path.join(Server_path)
        if not os.path.exists(S_weights_path):
            os.makedirs(S_weights_path)

        with open(os.path.join(S_weights_path, f"BS_{BS_ID}_weights_round_{rnd}.obj"), 'wb') as h:
            pickle.dump(weights_metrices, h)

        # Load global weights after the first round
        if rnd > 1:
            global_weights_path = os.path.join(gserver_path, "Global_weights.obj")
            with open(global_weights_path, 'rb') as h:
                aggregated_weights = pickle.load(h)

        return aggregated_weights, aggregated_metrics


# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = 'intermediate_server2:5004', 
        config=fl.server.ServerConfig(num_rounds=3) ,
        strategy = strategy
)