import argparse
import sys
import threading
import io
import gzip
import time
import os
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Dict, Any, List
import torch
import uvicorn
from model import SimpleNet

app = FastAPI()

# Global state for Parameter Server
# Global state for Parameter Server
global_model = SimpleNet()
optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01, momentum=0.9)
model_version = 0
gradient_buffer = []
BUFFER_SIZE = 2
TARGET_VERSIONS = 10
SERVER_PIN = None

# Concurrency safety: Lock for endpoints that modify or read consistent state
lock = threading.Lock()

# Metrics tracking
training_start_time = None
total_bytes_received = 0

def verify_pin(x_auth_pin: str = Header(None)):
    """Dependency to check the authentication PIN."""
    if x_auth_pin != SERVER_PIN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid PIN")
    return x_auth_pin

@app.get("/version")
def get_version(pin: str = Depends(verify_pin)):
    """Returns the current model version."""
    with lock:
        return {"version": model_version}

@app.get("/model")
def get_model(pin: str = Depends(verify_pin)):
    """Returns the current model weights and version."""
    with lock:
        # Convert tensors to python lists for JSON serialization
        state_dict = {
            k: v.cpu().tolist() for k, v in global_model.state_dict().items()
        }
        return {"version": model_version, "weights": state_dict}

@app.post("/submit_gradients")
async def submit_gradients(request: Request, pin: str = Depends(verify_pin)):
    """
    Receives compressed binary gradients from workers, buffers them until BUFFER_SIZE is reached,
    then averages them, updates the global model weights safely, and increments version.
    """
    global model_version
    global training_start_time
    global total_bytes_received
    
    if training_start_time is None:
        training_start_time = time.time()
    
    worker_id = request.headers.get("X-Worker-Id", "unknown")
    
    try:
        body = await request.body()
        total_bytes_received += len(body)
        decompressed_data = gzip.decompress(body)
        buffer = io.BytesIO(decompressed_data)
        # We use weights_only=True to safely load the binary tensor payload buffer
        worker_grads = torch.load(buffer, map_location="cpu", weights_only=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode binary payload: {e}")
        
    with lock:
        gradient_buffer.append(worker_grads)
        print(f" [Server] Received gradients from Worker {worker_id}. Buffer: {len(gradient_buffer)}/{BUFFER_SIZE}")
        
        # Check if we have received exactly BUFFER_SIZE (2) gradient submissions
        if len(gradient_buffer) == BUFFER_SIZE:
            print(f"⚙️ [Server] Buffer full! Averaging gradients from {BUFFER_SIZE} workers...")
            # Average the gradients
            avg_grads = {}
            for name in gradient_buffer[0].keys():
                # Stack all worker gradient tensors for the same parameter name
                stacked = torch.stack([w_grads[name] for w_grads in gradient_buffer], dim=0)
                # Compute the mean across workers
                avg_grads[name] = torch.mean(stacked, dim=0)

            # PyTorch In-Place Update Fix: Use a real optimizer instead of manual subtraction.
            # Manual subtraction causes unstable training, exploding gradients, and 10% accuracy.
            optimizer.zero_grad()
            for name, param in global_model.named_parameters():
                if name in avg_grads:
                    # Inject the averaged gradients directly into the global model's .grad attribute
                    param.grad = avg_grads[name]
                    
            # Let PyTorch's native SGD (with momentum) safely update the parameters
            optimizer.step()
                    
            # Increment the version and clear the gradient buffer safely
            model_version += 1
            print(f"++++[Server] Global Model updated to Version {model_version}")
            gradient_buffer.clear()
            
            # Save the model gracefully once it hits TARGET_VERSIONS
            if model_version >= TARGET_VERSIONS:
                training_duration = time.time() - training_start_time
                mb_received = total_bytes_received / (1024 * 1024)
                
                print("\n" + "-"*40)
                print(" TRAINING SESSION METADATA")
                print("-"*40)
                print(f" Total Duration      : {training_duration:.2f} seconds")
                print(f" Total Data Received : {mb_received:.4f} MB")
                print(f" Global Epochs       : {TARGET_VERSIONS}")
                print(f" Worker Count        : {BUFFER_SIZE}")
                print("-"*40)
                
                torch.save(global_model.state_dict(), "trained_model.pth")
                print(" Saved weights to 'trained_model.pth'\n")
                
                # Forcefully shutdown the server so it stops responding to ghost/lagging workers
                print("🏁 Training complete. Shutting down server...")
                os._exit(0)
            
            return {
                "status": "success", 
                "message": "Model updated", 
                "new_version": model_version
            }
            
    return {"status": "success", "message": "Gradients buffered"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Server")
    parser.add_argument("--pin", type=str, help="4-digit PIN for authentication")
    args = parser.parse_args()

    # Dynamic PIN logic
    if args.pin:
        SERVER_PIN = args.pin
    else:
        while True:
            pin_input = input("Set a 4-digit PIN for the server: ")
            if len(pin_input) == 4 and pin_input.isdigit():
                SERVER_PIN = pin_input
                break
            print("Invalid input. Please enter exactly 4 digits.")
            
    print("\n--- Server Configuration ---")
    while True:
        try:
            BUFFER_SIZE = int(input("Enter WORLD_SIZE (Number of workers to wait for, e.g., 2): "))
            TARGET_VERSIONS = int(input("Enter TOTAL_GLOBAL_BATCHES (Number of global averages to perform, e.g., 50): "))
            if BUFFER_SIZE > 0 and TARGET_VERSIONS > 0:
                break
            print("Please enter positive integers.")
        except ValueError:
            print("Please enter valid integers.")
    
    print(f" Server secured with PIN: {SERVER_PIN} | World Size: {BUFFER_SIZE} | Target Epochs: {TARGET_VERSIONS}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
