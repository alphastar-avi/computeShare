# ComputeShare

ComputeShare is a lightweight federated machine learning training system with pytorch. It enables users to distribute PyTorch training tasks across multiple Macs over the internet, utilizing native MPS (Metal Performance Shaders) acceleration to split the workload and drastically reduce training time.

## Architecture

The system consists of two primary components operating in a **Bulk Synchronous Parallel** formation:
1. **Parameter Server (`server.py`)**: The central node that holds the global PyTorch model. It asynchronously waits for workers to submit their computed gradients, decompresses the payload, mathematically averages them via SGD, and updates the global weights. The server implements the **Linear Scaling Rule**, dynamically multiplying the learning rate by the total Worker `WORLD_SIZE` to prevent mathematical decay in multi-node clusters.
2. **Workers (`worker.py`)**: Distributed clients that pull the latest model from the server, process a unique mathematical shard of the training dataset using their local GPU, and submit the calculated vectors back to the server.

### Safeties & Constraints
- **Stale Gradients Rejection**: To prevent a slow worker from polluting the global weights, workers attach an `X-Worker-Version` HTTP header with their gradients. If the Server has already advanced to a new global version, it immediately mathematically rejects the slow worker's payload (`HTTP 409 Conflict`) and forces the worker to re-pull the new weights and recompute.
- **Connection Continuity**: The workers leverage extended `requests` timeouts (15s/30s) to survive aggressive LocalTunnel connection spikes without experiencing process-ending timeout drops.
- **Compression**: To drastically decrease network overhead (e.g., preventing Ngrok blocks), gradients are **not** serialized into JSON. The workers serialize their PyTorch tensors using `io.BytesIO()`, compress them locally via `gzip`, and transmit the binary payload (`application/octet-stream`).
- **Authentication**: All external HTTP communication is secured with a mandatory 4-digit PIN header.

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
python server.py --dataset MNIST --pin 1234
```

**Inline Initialization:**
```bash
python server.py --dataset MNIST --pinSizEpo <PIN> <WORLD_SIZE> <TOTAL_GLOBAL_BATCHES>
# Example: python server.py --dataset FashionMNIST --pinSizEpo 1234 2 50
```
You will be prompted to define:
- `WORLD_SIZE`: Total number of distributed workers.
- `TOTAL_GLOBAL_BATCHES`: The number of gradient averaging rounds before the model is saved.
- `--dataset`: The dataset to use (e.g., `MNIST`, `FashionMNIST`, `CIFAR10`). Default is `MNIST`.

*Note: For external connections over the internet, we highly recommend exposing port 8000 using a Free Cloudflare Tunnel for maximum speed and stability:*
`npx cloudflared tunnel --url http://localhost:8000`
*(Alternatively, if all Macs are on the exact same Wi-Fi/Shared Network, simply use the server's Local IP Address `http://192.168.x.x:8000` natively without any tunnels for zero-latency setups!)*

### 2. Connect the Workers
On any machine participating in the training, execute the worker script.

**Interactive Initialization:**
```bash
python worker.py --dataset MNIST --pin 1234
```

**Inline Initialization:**
```bash
python worker.py --dataset MNIST --pinSizRanBatEpo <PIN> <WORLD_SIZE> <RANK> <BATCH_SIZE> <TOTAL_GLOBAL_BATCHES>
# Example: python worker.py --dataset FashionMNIST --pinSizRanBatEpo 1234 2 0 32 50
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
python test.py --dataset MNIST
```

### Supported Datasets (`--dataset`)
Any recognized `torchvision` dataset works dynamically without crashing, thanks to the **Universal Dataset Factory** in `utils.py`. The Factory acts as an intelligent API router that automatically resolves PyTorch's wildly inconsistent `train=True` vs `split='train'` kwargs, maps all topological class bounds so the MPS/Cuda device dynamically scales its final layers, and securely rejects multi-label datasets like *CelebA* before they initialize. 

Furthermore, the Factory features a **Dynamic Auto-Installer** that actively intercepts missing underlying dataset dependencies (e.g., PyTorch crashing because it needs `h5py` or `gdown` for PCAM) and iteratively installs them on the fly in the background via `pip`! `test.py` also no longer hardcodes "10,000" and will natively calculate the precise population volume of any validation shape processed. 

**Ranked Performance Index**:
*(Keep in mind, to keep bandwidth extremely low (< 50KB/sec), all input is mathematically shrunk to 28x28 grayscale tensors).*

**1. High Performance Tier** *(95%+ Accuracy natively - Simple contours natively scale)*:
- `MNIST` (Handwritten digits)
- `FashionMNIST` (Articles of clothing)
- `KMNIST` (Kuzushiji characters)
- `QMNIST` (Extended Handwritten Digits)
- `EMNIST` (Extended digits/letters)
- `USPS` (Postal Digits)

**2. Medium Performance Tier** *(60% - 80% Accuracy natively - Silhouettes survive thresholding)*:
- `CIFAR10` (10 classes of objects)
- `SVHN` (Street View House Numbers)
- `STL10` (Higher Res Objects)

**3. Low Performance Tier** *(Runs flawlessly over the network, but suffers deep predictive accuracy penalties when shrunk to 28x28 Grayscales due to heavy reliance on High-Resolution Color mapping)*:
- `CIFAR100` (100 classes of objects)
- `StanfordCars` (Car Models - 196 Classes)
- `PCAM` (Medical Cancer scans)
- `EuroSAT` (Satellite Imagery - 10 Classes)
- `Flowers102` (Flower Species - 102 Classes)
- `OxfordIIITPet` (Pets - 37 Breeds)
- `Places365` (Scenes/Places - 365 Classes)
- `Food101` (Food Dishes - 101 Classes)
- `GTSRB` (Traffic Signs - 43 Classes)
- `DTD` (Textures - 47 Classes)
- `FGVCAircraft` (Aircraft Models - 100 Classes)
- `Country211` (Photos by Country - 211 Classes)
- `Caltech101` / `Caltech256` (General Objects)
