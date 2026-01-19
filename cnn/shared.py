from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
current_dir = Path(__file__).parent
project_dir = current_dir.parent
DATA_DIR = project_dir / "data"


def make_model(resnet=False, num_classes: int = 10):
    model = MCNN15(num_classes=num_classes)
    return model, current_dir / "fashion_cnn_best.pt"


class MCNN15(nn.Module):
    def __init__(self, num_classes=10):
        super(MCNN15, self).__init__()

        # Helper function to create a Conv-BatchNorm-ReLU block
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            # --- Convolutional Layer Group 1 ---
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 64),
            conv_block(64, 32),
            conv_block(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 14 x 14
            # --- Convolutional Layer Group 2 ---
            conv_block(64, 256),
            conv_block(256, 192),
            conv_block(192, 128),
            conv_block(128, 64),
            conv_block(64, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 7 x 7
            # --- Convolutional Layer Group 3 ---
            conv_block(32, 256),
            conv_block(256, 256),
            conv_block(256, 256),
            conv_block(256, 128),
            conv_block(128, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 3 x 3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Input features = 32 channels * 3 * 3 spatial dim = 288
            nn.Linear(32 * 3 * 3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_resnet(num_classes: int = 10):
    model = torchvision.models.resnet50(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_loaders(batch_size: int = 128, root=DATA_DIR, use_workers=True):
    train_ds = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(20),
            ]
        ),
    )
    test_ds = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transforms.ToTensor()
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
