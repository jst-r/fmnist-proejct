from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = Path(__file__).parent
project_dir = current_dir.parent
DATA_DIR = project_dir / "data"


def make_model(resnet=False, num_classes: int = 10):
    model = nn.Sequential(
        # 1. Input Padding: Paper uses 32x32 input for 28x28 images
        nn.ZeroPad2d(2),
        # 2. First Layer (C1): Conv 5x5, 6 filters, stride 1
        # Output: 28x28x6
        nn.Conv2d(1, 6, kernel_size=5, stride=1),
        nn.ReLU(),
        # 3. Second Layer (S2): Avg Pooling 2x2, stride 2
        # Output: 14x14x6
        nn.AvgPool2d(kernel_size=2, stride=2),
        # 4. Third Layer (C3): Conv 5x5, 16 filters, stride 1
        # Output: 10x10x16
        nn.Conv2d(6, 16, kernel_size=5, stride=1),
        nn.ReLU(),
        # 5. Fourth Layer (S4): Avg Pooling 2x2, stride 2
        # Output: 5x5x16
        nn.AvgPool2d(kernel_size=2, stride=2),
        # 6. Flatten for Dense Layers
        nn.Flatten(),
        # 7. Fifth Layer (C5/FC1): Fully connected 120 units
        nn.Linear(16 * 5 * 5, 120),
        nn.ReLU(),
        # 8. Sixth Layer (F6/FC2): Fully connected 84 units
        nn.Linear(120, 84),
        nn.ReLU(),
        # 9. Output Layer: Fully connected 10 units (Softmax)
        nn.Linear(84, 10),
        # Note: nn.CrossEntropyLoss in PyTorch applies Softmax internally
    )
    return model, current_dir / "fashion_cnn_best.pt"


def make_resnet(num_classes: int = 10):
    model = torchvision.models.resnet50(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_loaders(batch_size: int = 128, root=DATA_DIR, use_workers=True):
    transform = transforms.ToTensor()

    train_ds = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_ds = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )

    num_workers = 2 if use_workers else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
