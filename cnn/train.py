import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared import device, get_loaders, make_model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return loss_sum / total, correct / total


def main():
    train_loader, test_loader = get_loaders(batch_size=128)

    model, save_path = make_model(resnet=False)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = 25

    test_acc = 0.0
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        end_time = time.time()
        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"time={end_time - start_time:.2f}"
        )
        if test_acc > best_acc:
            best_acc = test_acc

    torch.save(model.state_dict(), save_path)

    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
