import argparse
import sys
import time
import requests
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model import SimpleNet

# ==========================================
# USER CONFIGURATION
# ==========================================
# Define your universal Model and torchvision Dataset here.
# The system will automatically shard this data among all workers.

model = SimpleNet()

# We use MNIST here as a standard torchvision dataset.
# The transform converts PIL images to Tensors so the model can process them.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# ==========================================

SERVER_URL = "http://localhost:8000"
WORKER_PIN = None

def get_headers():
    return {
        "X-Auth-Pin": WORKER_PIN,
        "ngrok-skip-browser-warning": "true"
    }

def get_server_version():
    """Queries the parameter server for the current model version."""
    try:
        response = requests.get(f"{SERVER_URL}/version", headers=get_headers())
        response.raise_for_status()
        return response.json()["version"]
    except Exception as e:
        print(f"Failed to get version: {e}")
        return None

def pull_model(model):
    """Pulls the latest weights and version from the parameter server."""
    try:
        response = requests.get(f"{SERVER_URL}/model", headers=get_headers())
        response.raise_for_status()
        data = response.json()
        version = data["version"]
        weights = data["weights"]
        
        # Convert Lists back to PyTorch Tensors and load state dict
        state_dict = {k: torch.tensor(v) for k, v in weights.items()}
        model.load_state_dict(state_dict)
        return version
    except Exception as e:
        print(f"Failed to pull model: {e}")
        return None

def submit_gradients(worker_id, grads):
    """Submits the computed gradients to the parameter server in JSON-friendly format."""
    # Convert grad tensors to standard Python lists for JSON serialization
    grads_list = {k: v.cpu().tolist() for k, v in grads.items() if v is not None}
    
    payload = {
        "worker_id": worker_id,
        "grads": grads_list
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/submit_gradients", json=payload, headers=get_headers())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to submit gradients: {e}")
        return None

def main(world_size: int, rank: int, batch_size: int, target_versions: int, worker_id: str):
    print(f"\n🚀 Worker {worker_id} (Rank {rank}/{world_size-1}) starting...")
    
    # ---------------------------------------------------------
    # Universal Dataset Sharding (Data Parallelism)
    # ---------------------------------------------------------
    total_samples = len(dataset)
    
    # Generate a list of all indices in the dataset
    all_indices = list(range(total_samples))
    
    # Slice the indices. This worker takes every Nth sample starting from its RANK.
    # Ex: World_size=2. Rank 0 gets [0, 2, 4, 6...]. Rank 1 gets [1, 3, 5, 7...]
    worker_indices = all_indices[rank :: world_size]
    
    # Create the subset specifically for this worker
    worker_subset = Subset(dataset, worker_indices)
    
    # ---------------------------------------------------------
    # Performance Optimization for macOS Apple Silicon
    # ---------------------------------------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Hardware Acceleration: Running PyTorch on {device.type.upper()}")
    
    # Move the model to the GPU/MPS
    model.to(device)

    # Load the subset into a DataLoader
    # pin_memory throws a warning on MPS, so it has been removed.
    dataloader = DataLoader(worker_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"📊 Dataset successfully sharded. This worker gets {len(worker_subset)}/{total_samples} samples.")
    print("---------------------------------------------------------")
    
    last_trained_version = -1
    
    # Core Distributed Loop over the dataloader bounds
    for batch_idx, (data, target) in enumerate(dataloader):
        
        while True:
            try:
                # Polling phase: Check server's current version
                current_version = get_server_version()
                if current_version is None:
                    time.sleep(2)
                    continue
                    
                # Crucial Synchronization Logic: 
                # If the version hasn't incremented since our last training step, DO NOT process the batch.
                if current_version == last_trained_version:
                    time.sleep(0.5)
                    continue
                    
                # Pull new target weights for this batch
                version = pull_model(model)
                if version is None:
                    time.sleep(2)
                    continue
                
                # Move data and targets to the same device as the model (MPS)
                data, target = data.to(device), target.to(device)
                
                # Forward Pass
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                # Backward Pass
                model.zero_grad()
                loss.backward()
                
                # Extract gradients to standard Python objects natively
                grads = {name: param.grad for name, param in model.named_parameters()}
                
                # Submit computed gradients to the Parameter Server
                submit_gradients(worker_id, grads)
                print(f"[Batch {batch_idx+1}] Worker {worker_id} submitted gradients for Server Version {version}. Loss: {loss.item():.4f}")
                
                # Record successful train step to prevent double-dipping the same weights
                last_trained_version = version
                time.sleep(1.0) # Pace for free-tier loc tunnel
                
                # Break the polling loop and move to the next batch of images
                break

            except ConnectionError:
                print("Could not connect to server, waiting...")
                time.sleep(2)
            except Exception as e:
                print(f"Error during training loop: {e}")
                time.sleep(2)
                
        # Check if we've successfully computed enough batches
        if last_trained_version >= target_versions:
            print(f"\n⏳ Worker {worker_id} finished computing its {target_versions} global batches.")
            print("Waiting for slower workers to finish so the server can combine them...")
            
            # The fast worker must stay alive and poll the server until the server 
            # confirms the final global sync is complete, otherwise the slow worker hangs.
            while True:
                final_version = get_server_version()
                if final_version is not None and final_version >= target_versions:
                    print(f"\n✅ Server confirmed {target_versions} global epochs complete! Shutting down gracefully.")
                    return # Exit the main function completely
                time.sleep(1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Worker")
    parser.add_argument("--pin", type=str, help="4-digit PIN for server authentication")
    args = parser.parse_args()

    # 1. Authenticaton Logic
    if args.pin:
        WORKER_PIN = args.pin
    else:
        while True:
            pin_input = input("🔑 Enter the 4-digit server PIN: ")
            if len(pin_input) == 4 and pin_input.isdigit():
                WORKER_PIN = pin_input
                break
            print("Invalid input. Please enter exactly 4 digits.")
            
    # 2. Universal Sharding Info Request
    print("\n--- Distributed Setup ---")
    while True:
        try:
            world_size = int(input("Enter WORLD_SIZE (Total number of workers, e.g., 2): "))
            rank = int(input("Enter RANK (This worker's ID, starting from 0): "))
            batch_size = int(input("Enter BATCH_SIZE (e.g., 16 or 32): "))
            target_versions = int(input("Enter TOTAL_EPOCHS (Must match server, e.g., 2): "))
            if rank >= world_size or rank < 0:
                print("Invalid configuration. RANK must be between 0 and (WORLD_SIZE - 1).")
                continue
            if batch_size <= 0 or target_versions <= 0:
                print("Batch size and epochs must be positive.")
                continue
            break
        except ValueError:
            print("Please enter valid integers.")

    # Generate a unique 8-character ID for tracking this specific worker process
    import uuid
    worker_id = str(uuid.uuid4())[:8]

    # Execute universal loop
    main(world_size=world_size, rank=rank, batch_size=batch_size, target_versions=target_versions, worker_id=worker_id)
