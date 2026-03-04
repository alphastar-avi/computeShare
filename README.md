# ComputeShare: Distributed Mac Training

ComputeShare is a lightweight federated machine learning system built for Apple Silicon. It enables users to distribute PyTorch training tasks across multiple Macs over the internet, utilizing native MPS (Metal Performance Shaders) acceleration to split the workload and drastically reduce training time.

## Architecture

The system consists of two primary components:
1. **Parameter Server (`server.py`)**: The central node that holds the global PyTorch model. It asynchronously waits for workers to submit their computed gradients, decompresses the payload, mathematically averages them via SGD, and updates the global weights.
2. **Workers (`worker.py`)**: Distributed clients that pull the latest model from the server, process a unique mathematical shard of the training dataset using their local GPU, and submit the calculated vectors back to the server.

To drastically decrease network overhead (e.g., preventing Ngrok blocks), gradients are **not** serialized into JSON. The workers serialize their PyTorch tensors `io.BytesIO()`, compress them locally using `gzip`, and transmit the binary payload (`application/octet-stream`) via HTTP to the parameter server. 

All external HTTP communication is secured with a mandatory 4-digit PIN header.

## Setup

Before running the system, initialize the Python environment and install the required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage Guide

### 1. Initialize the Parameter Server
Designate one machine to act as the parameter server. The server can be initialized interactively or via inline arguments to bypass prompts.

**Interactive Initialization:**
```bash
python server.py --pin 1234
```

**Inline Initialization:**
```bash
python server.py --pinSizEpo <PIN> <WORLD_SIZE> <TOTAL_GLOBAL_BATCHES>
# Example: python server.py --pinSizEpo 1234 2 50
```
You will be prompted to define:
- `WORLD_SIZE`: Total number of distributed workers.
- `TOTAL_GLOBAL_BATCHES`: The number of gradient averaging rounds before the model is saved.

*Note: For external connections, expose port 8000 using LocalTunnel:* 
`npx -y localtunnel --port 8000`

### 2. Connect the Workers
On any machine participating in the training, execute the worker script.

**Interactive Initialization:**
```bash
python worker.py --pin 1234
```

**Inline Initialization:**
```bash
python worker.py --pinSizRanBatEpo <PIN> <WORLD_SIZE> <RANK> <BATCH_SIZE> <TOTAL_GLOBAL_BATCHES>
# Example: python worker.py --pinSizRanBatEpo 1234 2 0 32 50
```

Regardless of initialization method, workers require:
- `WORLD_SIZE`: Must match the server configuration.
- `RANK`: The worker's unique ID (0 to `WORLD_SIZE - 1`).
- `BATCH_SIZE`: Number of images to process per forward pass.
- `TOTAL_GLOBAL_BATCHES`: Must match the server configuration.

*Note: Edit `SERVER_URL` on line 30 of `worker.py` if connecting via LocalTunnel.*

### 3. Evaluate the Model
Once the server reaches the target global batches, it will save the final weights to `trained_model.pth`. Execute the testing script to evaluate its accuracy on 10,000 novel images:
```bash
python test.py
```
