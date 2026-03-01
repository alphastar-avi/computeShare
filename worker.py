import argparse
import sys
import time
import requests
import torch
import torch.nn.functional as F
import uuid
from model import SimpleNet

SERVER_URL = "http://localhost:8000"
WORKER_PIN = None

def get_headers():
    return {"X-Auth-Pin": WORKER_PIN}

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

def main():
    worker_id = str(uuid.uuid4())[:8]
    print(f"Worker {worker_id} starting...")
    
    model = SimpleNet()
    last_trained_version = -1
    
    while True:
        try:
            # Check server's current version
            current_version = get_server_version()
            if current_version is None:
                time.sleep(2)
                continue
                
            # Worker Synchronization: If the version hasn't changed since last pull, sleep and poll again
            if current_version == last_trained_version:
                print(f"Version {current_version} unchanged. Waiting (time.sleep(0.5))...")
                time.sleep(0.5)
                continue
                
            # Pull new model
            version = pull_model(model)
            if version is None:
                time.sleep(2)
                continue
                
            print(f"Worker {worker_id} pulled model version {version}. Training...")
            
            # Generate dummy image-like data and targets for 10 classes
            x = torch.randn(16, 1, 28, 28)
            target = torch.randint(0, 10, (16,))
            
            # Forward pass
            output = model(x)
            loss = F.cross_entropy(output, target)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Extract gradients mapping parameter name to the Tensor grad
            grads = {name: param.grad for name, param in model.named_parameters()}
            
            # Submit to the Parameter Server
            submit_gradients(worker_id, grads)
            print(f"Worker {worker_id} submitted gradients for version {version}. Loss: {loss.item():.4f}")
            
            # Record that we've successfully trained on this version
            last_trained_version = version
            time.sleep(1.0) # Larger sleep to ensure free-tier ngrok doesn't drop
            
            # Bounding the training length for testing
            if last_trained_version >= 10:
                print(f"Worker {worker_id} reached 10 epochs. Terminating gracefully.")
                break

            
        except ConnectionError:
            print("Could not connect to server, waiting...")
            time.sleep(2)
        except Exception as e:
            print(f"Error during training loop: {e}")
            time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Worker")
    parser.add_argument("--pin", type=str, help="4-digit PIN for server authentication")
    args = parser.parse_args()

    if args.pin:
        WORKER_PIN = args.pin
    else:
        while True:
            pin_input = input("Enter the 4-digit server PIN: ")
            if len(pin_input) == 4 and pin_input.isdigit():
                WORKER_PIN = pin_input
                break
            print("Invalid input. Please enter exactly 4 digits.")
            
    main()
