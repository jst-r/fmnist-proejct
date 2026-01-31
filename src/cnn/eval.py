# %%
import time
import numpy as np
import torch

from src.cnn.shared import get_loaders, make_model, device
from src.common.eval import (
    print_classification_metrics,
    plot_confusion_matrix,
    plot_misclassified,
)
from src.common.paths import PROJECT_DIR


print("=" * 60)
print("FASHION-MNIST CNN EVALUATION")
print("=" * 60)

total_start = time.time()

# %%
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

print("\n[MODEL LOADING]")
load_start = time.time()
model, save_path = make_model()
model.to(device)
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()
print(f"[{time.time() - total_start:.2f}s] Model loaded from {save_path}")

# %%
print("\n[PREDICTION]")
pred_start = time.time()


def get_predictions(train: bool):
    pred_start = time.time()
    train_loader, eval_loader, test_loader = get_loaders(
        batch_size=512, use_workers=False
    )
    loader = train_loader if train else test_loader

    all_preds = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_images.append(imgs.cpu().numpy())

    elapsed = time.time() - pred_start
    print(
        f"[{time.time() - total_start:.2f}s] Generated predictions for {'train' if train else 'test'} set in {elapsed:.2f}s"
    )

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_images),
    )


# pred_train, y_train, x_train = get_predictions(train=True)
pred_test, y_test, x_test = get_predictions(train=False)

print("\n[RESULTS]")
print_classification_metrics(y_test, pred_test, class_names)

# %% Plot confusion matrix
plot_confusion_matrix(
    y_test, pred_test, class_names, PROJECT_DIR / "plots/cnn/confusion.png"
)
# %% Visualize misclassified examples
print("\n[MISCLASSIFIED EXAMPLES]")
plot_misclassified(
    x_test,
    y_test,
    pred_test,
    class_names,
    PROJECT_DIR / "plots/cnn/misclassified.png",
)

print(f"\n[{time.time() - total_start:.2f}s] Evaluation complete")
# %%
