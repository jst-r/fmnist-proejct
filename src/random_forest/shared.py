"""Shared utilities for Random Forest classifier."""

import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier

from src.common.paths import DATA_DIR

current_dir = Path(__file__).parent
MODEL_PATH = current_dir / "random_forest_model.pkl"


def load_dataset(train: bool):
    """Load Fashion-MNIST dataset and flatten images for Random Forest."""
    print(f"Loading {'train' if train else 'test'} dataset...")
    dataset = torchvision.datasets.FashionMNIST(
        DATA_DIR,
        train=train,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)

    Xs = []
    ys = []
    for imgs, labels in loader:
        Xs.append(imgs.view(imgs.size(0), -1).numpy())
        ys.append(labels.numpy())

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def make_model(n_estimators=100, criterion="entropy", max_depth=100, n_jobs=-1):
    """Create and return a Random Forest classifier."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        n_jobs=n_jobs,
    )
    return clf, MODEL_PATH


def save_model(clf, path):
    """Save a trained Random Forest model to disk."""
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {path}")


def load_model(path):
    """Load a trained Random Forest model from disk."""
    with open(path, "rb") as f:
        clf = pickle.load(f)
    print(f"Model loaded from {path}")
    return clf
