import flwr as fl
import sys
import numpy as np
import os
import pickle
# Retrieve environment variables
Server_path= os.environ.get("Server_path")
BS_ID = int(os.environ.get("BS_ID", 3))
gserver_path=os.environ.get("Gserver_path")

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        aggregated_weights, aggregated_metrics = aggregated_result
        S_weights_path=os.path.join(Server_path)
        if not os.path.exists(S_weights_path):
                os.makedirs(S_weights_path)
        h = open(os.path.join(S_weights_path, "BS_"+str(BS_ID)+"_weights_round_"+str(rnd)+".obj"), 'wb')
        pickle.dump(aggregated_weights, h)
        h.close()
        ################################################
        if rnd>1:
            global_weights_path = os.path.join(gserver_path, "Global_weights.obj")
            with open(global_weights_path, 'rb') as h:
                aggregated_weights = pickle.load(h)
        #################################################
        return aggregated_weights, aggregated_metrics

# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = 'intermediate_server1:5003', 
        config=fl.server.ServerConfig(num_rounds=3) ,
        strategy = strategy
)