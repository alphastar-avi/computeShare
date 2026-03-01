import threading
from fastapi import FastAPI
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

# Concurrency safety: Lock for endpoints that modify or read consistent state
lock = threading.Lock()

class Gradients(BaseModel):
    worker_id: str
    grads: Dict[str, list]  # Enforces standard Python lists from workers

@app.get("/version")
def get_version():
    """Returns the current model version."""
    with lock:
        return {"version": model_version}

@app.get("/model")
def get_model():
    """Returns the current model weights and version."""
    with lock:
        # Convert tensors to python lists for JSON serialization
        state_dict = {
            k: v.cpu().tolist() for k, v in global_model.state_dict().items()
        }
        return {"version": model_version, "weights": state_dict}

@app.post("/submit_gradients")
def submit_gradients(data: Gradients):
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
            
            # Save the model gracefully once it hits 10 epochs
            if model_version == 10:
                torch.save(global_model.state_dict(), "trained_model.pth")
                print("\n🎉 Training Complete! Saved weights to 'trained_model.pth'\n")
            
            return {
                "status": "success", 
                "message": "Model updated", 
                "new_version": model_version
            }
            
    return {"status": "success", "message": "Gradients buffered"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
