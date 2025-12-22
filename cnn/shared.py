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
    if resnet:
        return make_resnet(num_classes), current_dir / "fashion_resnet_best.pt"
    else:
        return make_model_custom(num_classes), current_dir / "fashion_cnn_best.pt"


def make_resnet(num_classes: int = 10):
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_model_custom(num_classes: int = 10):
    return nn.Sequential(
        nn.BatchNorm2d(1),
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(128 * 7 * 7, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(128, num_classes),
    )


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
