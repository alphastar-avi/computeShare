import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleNet

def evaluate_model():
    print("Loading test dataset (10,000 images)...")
    
    # We must use the exact same transforms we used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load the official MNIST "test" split
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset, batch_size=1000, shuffle=True)
    
    print("Loading your trained model weights...")
    model = SimpleNet()
    
    try:
        # Load the .pth file the server generated
        model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("\n Error: 'trained_model.pth' not found.")
        print("You need to successfully finish the 10-epoch distributed training with you and Roshini first!")
        return
        
    # IMPORTANT: Set the model to evaluation mode (turns off Dropout/BatchNorm layers if any exist)
    model.eval()
    
    correct = 0
    total = 0
    
    print("\n--- Running Inference ---")
    
    # torch.no_grad() tells PyTorch we are just taking a guess, NOT training.
    # It saves memory and makes the math much faster.
    with torch.no_grad():
        for images, labels in test_loader:
            # 1. Feed the images into your Federated Learning brain
            outputs = model(images)
            
            # 2. Which of the 10 classes got the highest probability?
            _, predicted = torch.max(outputs.data, 1)
            
            # 3. Tally up the score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"🎯 Total Accuracy on 10,000 never-before-seen images: {accuracy:.2f}%\n")
    
    # Let's inspect a few individual predictions directly
    print("--- Inspecting 5 Random Images ---")
    images, labels = next(iter(test_loader))
    
    for i in range(5):
        # We grab a single image and add a dummy "batch" dimension to it [1, 1, 28, 28]
        output = model(images[i].unsqueeze(0)) 
        _, pred = torch.max(output, 1)
        
        guess = pred.item()
        actual = labels[i].item()
        
        if guess == actual:
            status = "✅ Correct"
        else:
            status = "❌ Wrong"
            
        print(f"Image {i+1}: AI Guessed [{guess}] | Actual Answer [{actual}]  -> {status}")

if __name__ == "__main__":
    evaluate_model()