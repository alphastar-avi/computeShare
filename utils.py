import torch
import sys
import subprocess
import re
from torchvision import datasets, transforms

# Centralized Knowledge Base & Compatibility Map for Torchvision Datasets
DATASET_CONFIGS = {
    # --- HIGH PERFORMANCE (Native 28x28 Grayscale / Distinct Contours) ---
    'MNIST': {'classes': 10, 'train_arg': 'train', 'train_val': True, 'test_val': False},
    'FashionMNIST': {'classes': 10, 'train_arg': 'train', 'train_val': True, 'test_val': False},
    'KMNIST': {'classes': 10, 'train_arg': 'train', 'train_val': True, 'test_val': False},
    'QMNIST': {'classes': 10, 'train_arg': 'train', 'train_val': True, 'test_val': False},
    'EMNIST': {'classes': 62, 'train_arg': 'train', 'train_val': True, 'test_val': False, 'kwargs': {'split': 'byclass'}},
    'USPS': {'classes': 10, 'train_arg': 'train', 'train_val': True, 'test_val': False},

    # --- MEDIUM PERFORMANCE (Color/Downscaled but silhouettes survive thresholding) ---
    'CIFAR10': {'classes': 10, 'train_arg': 'train', 'train_val': True, 'test_val': False},
    'SVHN': {'classes': 10, 'train_arg': 'split', 'train_val': 'train', 'test_val': 'test'},
    'STL10': {'classes': 10, 'train_arg': 'split', 'train_val': 'train', 'test_val': 'test'},
    
    # --- LOW PERFORMANCE (Requires Heavy Color/High Resolution to distinguish 100+ classes) ---
    'CIFAR100': {'classes': 100, 'train_arg': 'train', 'train_val': True, 'test_val': False},
    'Caltech101': {'classes': 101, 'train_arg': 'None'}, # Standard implementation lacks native train test split arg directly available
    'Caltech256': {'classes': 256, 'train_arg': 'None'},
    'Flowers102': {'classes': 102, 'train_arg': 'split', 'train_val': 'train', 'test_val': 'test'},
    'Food101': {'classes': 101, 'train_arg': 'split', 'train_val': 'train', 'test_val': 'test'},
    'PCAM': {'classes': 2, 'train_arg': 'split', 'train_val': 'train', 'test_val': 'test'},
    'DTD': {'classes': 47, 'train_arg': 'split', 'train_val': 'train', 'test_val': 'test'},
    'GTSRB': {'classes': 43, 'train_arg': 'split', 'train_val': 'train', 'test_val': 'test'},
    'FER2013': {'classes': 7, 'train_arg': 'split', 'train_val': 'train', 'test_val': 'test'},
    'OxfordIIITPet': {'classes': 37, 'train_arg': 'split', 'train_val': 'trainval', 'test_val': 'test'},
    'StanfordCars': {'classes': 196, 'train_arg': 'split', 'train_val': 'train', 'test_val': 'test'},
    'Places365': {'classes': 365, 'train_arg': 'split', 'train_val': 'train', 'test_val': 'val'},
    'Country211': {'classes': 211, 'train_arg': 'split', 'train_val': 'train', 'test_val': 'test'},
    'FGVCAircraft': {'classes': 100, 'train_arg': 'split', 'train_val': 'train', 'test_val': 'test'},
    'EuroSAT': {'classes': 10, 'train_arg': 'none'}, # Satellite imagery (89MB)
    
    # --- INCOMPATIBLE Datasets (Multi-label instead of single integer Output) ---
    'CelebA': {'incompatible': 'Outputs 40 binary labels. Requires BCEWithLogitsLoss architecture.'}
}

def attempt_load_with_auto_install(dataset_class, kwargs, dataset_name):
    """
    Intelligently intercepts missing PyTorch dataset dependencies and auto-installs them via subprocess.
    Uses an iterative loop to handle cascading dependencies (e.g. PCAM requires both h5py AND gdown sequentially).
    """
    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        try:
            return dataset_class(**kwargs)
        except (RuntimeError, ModuleNotFoundError, ImportError) as e:
            error_msg = str(e)
            package_name = None
            
            # Scenario 1: PyTorch explicitly tells us to "pip install <X>"
            match = re.search(r"pip install ([\w\-]+)", error_msg)
            if match:
                package_name = match.group(1)
            # Scenario 2: Standard ModuleNotFoundError
            elif isinstance(e, ModuleNotFoundError):
                match = re.search(r"No module named '([\w\-]+)'", error_msg)
                if match:
                    package_name = match.group(1)
                    
            if package_name:
                print(f"\n[*] Missing dependency detected for {dataset_name}. Auto-installing '{package_name}' on the fly...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                    print(f"[*] Successfully installed '{package_name}'. Retrying dataset initialization...\n")
                    retries += 1
                    continue # Re-run the while loop with the newly injected site-packages!
                except subprocess.CalledProcessError:
                    raise RuntimeError(f"Failed to auto-install '{package_name}'. Please install it manually.")
                    
            # Re-raise if it's an unrelated mathematical error
            raise e

def get_dataset(dataset_name: str, root='./data', train=True, download=True):
    """
    Acts as a universal robust factory for Torchvision Datasets, intelligently rewriting parameters natively.
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if not hasattr(datasets, dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' not natively found in torchvision.datasets.")
        
    dataset_class = getattr(datasets, dataset_name)
    
    # 1. Base initialization arguments
    kwargs = {'root': root, 'download': download, 'transform': transform}
    
    # 2. Apply Custom Factory Overrides
    if dataset_name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[dataset_name]
        
        if 'incompatible' in config:
            raise NotImplementedError(f"'{dataset_name}' is structurally incompatible: {config['incompatible']}")
            
        # Merge extra internal kwargs (like EMNIST 'split')
        if 'kwargs' in config:
            kwargs.update(config['kwargs'])
            
        # Handle specialized Parameter Translation (train=True vs split='train')
        if config['train_arg'] == 'train':
            kwargs['train'] = config['train_val'] if train else config['test_val']
        elif config['train_arg'] == 'split':
            kwargs['split'] = config['train_val'] if train else config['test_val']
        elif config['train_arg'] == 'none':
            # Remove keys that the API doesn't support (e.g. EuroSAT)
            kwargs.pop('download', None) # EuroSAT doesn't take a download flag in some versions
            if 'train' in kwargs: del kwargs['train']

    # 3. Load the data using our auto-installer wrapper
    dataset = attempt_load_with_auto_install(dataset_class, kwargs, dataset_name)

    # 4. Handle Datasets that LACK a native train/test split (Deterministic 80/20)
    if dataset_name in DATASET_CONFIGS and DATASET_CONFIGS[dataset_name]['train_arg'] == 'none':
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        # Using a FIXED generator seed ensures all workers/testers see the same non-overlapping images
        train_ds, test_ds = torch.utils.data.random_split(
            dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
        )
        return train_ds if train else test_ds

    return dataset

def get_num_classes(dataset_name: str) -> int:
    """Dynamically resolves the EXACT output dimensionality needed for the network layer."""
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name].get('classes', 10)
    return 10
