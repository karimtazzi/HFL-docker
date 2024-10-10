import os
import pickle
import time
import json
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Input, Dropout,
    Bidirectional, Conv1D, MaxPooling1D,
    LSTM
)
import flwr as fl
from flwr.client import start_client

base_name="Hierarchical_FL/Base_station_"+str(sys.argv[1])
# Retrieve environment variables
BASE_STATION_PATH = os.environ.get("BASE_STATION_PATH", base_name)
CLIENT_ID = int(os.environ.get("CLIENT_ID", 3))

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

# Load data
client_path = os.path.join(BASE_STATION_PATH, f"Client_{CLIENT_ID}", f"Client_{CLIENT_ID}.csv")
DATA = pd.read_csv(client_path)
X = DATA.drop(['Label'], axis=1).values
y = DATA['Label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create the model
model = create_cnn_model(X_train_shape=X_train.shape[1])

# Define the Flower client
class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r=model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=2)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        hist = r.history
        print("Fit history:", hist)
        return model.get_weights(), len(X_train), {"accuracy": accuracy, "loss": loss}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
        y_pred = model.predict(X_test, verbose=2)

        y_pred_labels = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_test, y_pred_labels, average='weighted')
        recall = recall_score(y_test, y_pred_labels, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred_labels)
            
        results = {
            "client": CLIENT_ID,
            "accuracy": float(accuracy),
            "recall": recall,
            "f1": f1,
            "loss": loss
        }
        print(results)
        return loss, len(X_test), {"accuracy": accuracy, "recall": recall}

# Start Flower clisent
if __name__ == "__main__":
    server_address='intermediate_server'+str(sys.argv[1])+':'+str(sys.argv[2])
    start_client(server_address=server_address, client=Client().to_client())
