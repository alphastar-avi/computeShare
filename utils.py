import torch
from torchvision import datasets, transforms

def get_dataset(dataset_name: str, root='./data', train=True, download=True):
    """
    Dynamically loads a torchvision dataset and applies universal transforms
    so it fits the lightweight 1-channel 28x28 SimpleNet model.
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Dynamically fetch the dataset class from torchvision.datasets
    if not hasattr(datasets, dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' is not found in torchvision.datasets")
        
    dataset_class = getattr(datasets, dataset_name)
    
    return dataset_class(root=root, train=train, download=download, transform=transform)

def get_num_classes(dataset_name: str) -> int:
    """Returns the number of classes for common datasets, defaults to 10."""
    mapping = {
        'EMNIST': 62,
        'CIFAR100': 100,
    }
    return mapping.get(dataset_name, 10)
