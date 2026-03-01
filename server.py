import argparse
import sys
import threading
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List
import torch
import uvicorn
from model import SimpleNet

app = FastAPI()

# Global state for Parameter Server
global_model = SimpleNet()
model_version = 0
gradient_buffer = []
BUFFER_SIZE = 2
TARGET_VERSIONS = 10
SERVER_PIN = None

# Concurrency safety: Lock for endpoints that modify or read consistent state
lock = threading.Lock()

class Gradients(BaseModel):
    worker_id: str
    grads: Dict[str, list]  # Enforces standard Python lists from workers

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
def submit_gradients(data: Gradients, pin: str = Depends(verify_pin)):
    """
    Receives gradients from workers, buffers them until BUFFER_SIZE is reached,
    then averages them, updates the global model weights safely, and increments version.
    """
    global model_version
    with lock:
        gradient_buffer.append(data.grads)
        
        # Check if we have received exactly BUFFER_SIZE (2) gradient submissions
        if len(gradient_buffer) == BUFFER_SIZE:
            # Average the gradients
            avg_grads = {}
            for name in gradient_buffer[0].keys():
                # Stack all worker gradients for the same parameter name
                stacked = torch.tensor([worker_grads[name] for worker_grads in gradient_buffer])
                # Compute the mean across workers
                avg_grads[name] = torch.mean(stacked, dim=0)

            # PyTorch In-Place Update Fix: update the underlying tensor data directly
            learning_rate = 0.01
            for name, param in global_model.named_parameters():
                if name in avg_grads:
                    # Modify the underlying tensor data directly to avoid in-place modification errors
                    param.data -= learning_rate * avg_grads[name]
                    
            # Increment the version and clear the gradient buffer safely
            model_version += 1
            gradient_buffer.clear()
            
            # Save the model gracefully once it hits TARGET_VERSIONS
            if model_version >= TARGET_VERSIONS:
                torch.save(global_model.state_dict(), "trained_model.pth")
                print("\n Training Complete! Saved weights to 'trained_model.pth'\n")
            
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
            TARGET_VERSIONS = int(input("Enter TOTAL_EPOCHS (Number of global averages to perform, e.g., 2): "))
            if BUFFER_SIZE > 0 and TARGET_VERSIONS > 0:
                break
            print("Please enter positive integers.")
        except ValueError:
            print("Please enter valid integers.")
    
    print(f" Server secured with PIN: {SERVER_PIN} | World Size: {BUFFER_SIZE} | Target Epochs: {TARGET_VERSIONS}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
