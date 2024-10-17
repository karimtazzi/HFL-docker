# Hierarchical Federated Learning with Malicious Clients Detection (HFL-Docker)

**HFL-Docker** is a hierarchical federated learning system that incorporates malicious clients detection to improve security for IoT networks. This project leverages federated learning architecture with various nodes like smart meters, data concentrators, and a global server. The system is dockerized, allowing easy setup and deployment across distributed networks to mitigate potential IoT attacks.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Project](#running-the-project)
  - [Stopping the Project](#stopping-the-project)

## Overview

This project implements **Hierarchical Federated Learning (HFL)** for securing IoT networks, particularly focusing on detecting and mitigating malicious clients in a smart grid context. The HFL architecture is hierarchical, with **Smart Meters** sending local updates to **Data Concentrators**, which in turn forward aggregated updates to the **Global Server**. The system is designed to defend against adversarial attacks by identifying compromised clients (malicious actors).

## Project Structure

```bash
HFL-DOCKER/
├── DataConcentrator/                     # Data concentrator nodes in the hierarchy
│   ├── Dockerfile-dataConcentrator1      # Dockerfile for DataConcentrator1
│   ├── Dockerfile-dataConcentrator2      # Dockerfile for DataConcentrator2
│   ├── intermediate_server1.py           # First data concentrator script
│   ├── intermediate_server2.py           # Second data concentrator script
│   ├── requirements.txt                  # Dependencies for data concentrators
│
├── GlobalServer/                         # Global federated learning server
│   ├── Dockerfile-globalServer           # Dockerfile for the global server
│   ├── Server.py                         # Central server script
│   ├── requirements.txt                  # Dependencies for the global server
│
├── Hierarchical_FL/                      # Hierarchical federated learning implementation files
│
├── SmartMeters/                          # Smart meter clients that send data
│   ├── Client.py                         # Smart Meter client script
│   ├── Dockerfile-SmartMeters1           # Dockerfile for SmartMeter1
│   ├── Dockerfile-SmartMeters2           # Dockerfile for SmartMeter2
│   ├── Dockerfile-SmartMeters_Mal1       # Dockerfile for a malicious SmartMeter1
│   ├── Dockerfile-SmartMeters_Mal2       # Dockerfile for a malicious SmartMeter2
│   ├── MalClient.py                      # Malicious client implementation
│   ├── requirements.txt                  # Dependencies for smart meters
│
├── docker-compose.yml                    # Docker Compose file for setting up the environment
├── .gitignore                            # Git ignore file
└── README.md                             # Project documentation
```

## Prerequisites

Ensure the following software is installed on your system:

- [Docker](https://docs.docker.com/get-docker/) (v20.x.x or higher)
- [Docker Compose](https://docs.docker.com/compose/install/) (v1.29.x or higher)
- [Git](https://git-scm.com/)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/karimtazzi/HFL-docker.git
   ```

2. Navigate into the project directory:

   ```bash
   cd HFL-docker
   ```

3. Build the Docker images for the project:

   ```bash
   docker-compose build
   ```

## Usage

### Running the Project

To start the federated learning network (Global Server, Data Concentrators, Smart Meters, and Malicious Clients):

```bash
docker-compose up
```

This command will set up and run:

- **Global Server**: The main federated learning server.
- **Data Concentrators**: Intermediate servers for aggregating updates from smart meters.
- **Smart Meters**: IoT devices (clients) sending data updates to the system.
- **Malicious Clients**: Simulated compromised IoT devices for attack detection.

### Stopping the Project

To stop all running containers:

```bash
docker-compose down
```

This will stop and remove all containers associated with the federated learning network.
