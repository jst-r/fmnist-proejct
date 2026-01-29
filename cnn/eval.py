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


def get_predictions(train: bool, return_images: bool = False):
    pred_start = time.time()
    train_loader, eval_loader, test_loader = get_loaders(
        batch_size=512, use_workers=False
    )
    loader = train_loader if train else test_loader

    all_preds = []
    all_labels = []
    all_images = [] if return_images else None

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            if return_images:
                all_images.append(imgs.cpu().numpy())

    elapsed = time.time() - pred_start
    print(
        f"[{time.time() - total_start:.2f}s] Generated predictions for {'train' if train else 'test'} set in {elapsed:.2f}s"
    )

    if return_images:
        return (
            np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_images),
        )
    return np.concatenate(all_preds), np.concatenate(all_labels)


pred_train, y_train = get_predictions(train=True, return_images=False)
pred_test, y_test, x_test = get_predictions(train=False, return_images=True)

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

# %% Visualize misclassified examples
print("\n[MISCLASSIFIED EXAMPLES]")

# Find incorrect predictions
incorrect_mask = pred_test != y_test
incorrect_indices = np.where(incorrect_mask)[0]

# Group by (true_label, pred_label) pairs
pair_counts = {}
pair_indices = {}

for idx in incorrect_indices:
    true_label = y_test[idx]
    pred_label = pred_test[idx]
    pair = (true_label, pred_label)

    if pair not in pair_counts:
        pair_counts[pair] = 0
        pair_indices[pair] = []
    pair_counts[pair] += 1
    pair_indices[pair].append(idx)

# Get top 5 most common misclassification pairs
top_5_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]

print(f"Top 5 misclassification pairs:")
for (true_label, pred_label), count in top_5_pairs:
    print(f"  {class_names[true_label]} → {class_names[pred_label]}: {count} examples")

# Create figure with 5 rows x 9 columns
fig, axes = plt.subplots(5, 9, figsize=(18, 10))
fig.suptitle(
    "Top 5 Misclassification Pairs (Random 9 Samples Each)", fontsize=16, y=0.98
)

for row, ((true_label, pred_label), count) in enumerate(top_5_pairs):
    indices = pair_indices[(true_label, pred_label)]

    # Randomly sample 9 examples (or fewer if not enough)
    sample_size = min(9, len(indices))
    sampled_indices = np.random.choice(indices, size=sample_size, replace=False)

    # Fill the row
    for col in range(9):
        ax = axes[row, col]

        if col < sample_size:
            idx = sampled_indices[col]
            img = x_test[idx]

            # Convert from (C, H, W) to (H, W, C) for display
            if img.shape[0] == 1:  # Grayscale
                img = img.squeeze(0)
            else:
                img = np.transpose(img, (1, 2, 0))

            ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
            ax.axis("off")
        else:
            ax.axis("off")

    # Add row label on the left
    axes[row, 0].set_ylabel(
        f"{class_names[true_label]}\n→\n{class_names[pred_label]}",
        rotation=0,
        ha="right",
        va="center",
        fontsize=10,
        labelpad=20,
    )

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

print(f"\n[{time.time() - total_start:.2f}s] Evaluation complete")
# %%
