# %%
import time
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn.metrics
import torch

from shared import get_loaders, make_model, device


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

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    elapsed = time.time() - pred_start
    print(
        f"[{time.time() - total_start:.2f}s] Generated predictions for {'train' if train else 'test'} set in {elapsed:.2f}s"
    )
    return np.concatenate(all_preds), np.concatenate(all_labels)


pred_train, y_train = get_predictions(train=True)
pred_test, y_test = get_predictions(train=False)

print("\n[RESULTS]")
print("=" * 60)
print(f"Train accuracy:  {sklearn.metrics.accuracy_score(y_train, pred_train):.4f}")
print(f"Test accuracy:   {sklearn.metrics.accuracy_score(y_test, pred_test):.4f}")
print(
    f"Test precision:  {sklearn.metrics.precision_score(y_test, pred_test, average='weighted'):.4f}"
)
print(
    f"Test recall:     {sklearn.metrics.recall_score(y_test, pred_test, average='weighted'):.4f}"
)
print("=" * 60)

for label in range(10):
    print(f"Label: {class_names[label]}")
    print(
        "Precision:\t{:.4f}".format(
            sklearn.metrics.precision_score(
                y_test, pred_test, average="weighted", labels=[label]
            ),
        )
    )
    print(
        "Recall:\t\t{:.4f}".format(
            sklearn.metrics.recall_score(
                y_test, pred_test, average="weighted", labels=[label]
            ),
        )
    )

# %% Plot
confusion = sklearn.metrics.confusion_matrix(y_test, pred_test)
norm_confusion = confusion / confusion.sum(axis=1, keepdims=True) * 100
log_confusion = np.log1p(confusion)
sns.heatmap(
    log_confusion,
    annot=norm_confusion,
    cmap="viridis",
    cbar=False,
    square=True,
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# %%
