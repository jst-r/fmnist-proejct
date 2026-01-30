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
print("=" * 60)
# print(f"Train accuracy:  {sklearn.metrics.accuracy_score(y_train, pred_train):.4f}")
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

# Configuration
N_PAIRS = 9  # Number of top misclassification pairs to display
SAMPLES_PER_PAIR = 9  # Number of random samples per pair
IMG_SIZE_INCHES = 2  # Size of each 28x28 image in inches

# Find incorrect predictions
incorrect_mask = pred_test != y_test
total_incorrect = incorrect_mask.sum()
shown_incorrect = 0
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

# Get top N most common misclassification pairs
top_n_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:N_PAIRS]

print(f"Top {N_PAIRS} misclassification pairs:")
for (true_label, pred_label), count in top_n_pairs:
    shown_incorrect += count
    print(f"  {class_names[true_label]} → {class_names[pred_label]}: {count} examples")

print(
    f"Sampling {shown_incorrect} examples from {total_incorrect} total. {shown_incorrect / total_incorrect:.2%} of total"
)

# Create figure with N_PAIRS rows x (1 label column + SAMPLES_PER_PAIR image columns)
fig_width = IMG_SIZE_INCHES * (SAMPLES_PER_PAIR + 2)  # +2 for label column and spacing
fig_height = IMG_SIZE_INCHES * N_PAIRS
fig, axes = plt.subplots(N_PAIRS, SAMPLES_PER_PAIR + 1, figsize=(fig_width, fig_height))
fig.suptitle(
    f"Top {N_PAIRS} Misclassification Pairs ({SAMPLES_PER_PAIR} Random Samples Each)",
    fontsize=14,
    y=0.98,
)

for row, ((true_label, pred_label), count) in enumerate(top_n_pairs):
    indices = pair_indices[(true_label, pred_label)]

    # Randomly sample examples (or fewer if not enough)
    sample_size = min(SAMPLES_PER_PAIR, len(indices))
    sampled_indices = np.random.choice(indices, size=sample_size, replace=False)

    # First column: label
    label_ax = axes[row, 0]
    label_ax.text(
        0.5,
        0.5,
        f"True: {class_names[true_label]}\nPrediction: {class_names[pred_label]}",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        transform=label_ax.transAxes,
    )
    label_ax.axis("off")

    # Remaining columns: images
    for col in range(SAMPLES_PER_PAIR):
        ax = axes[row, col + 1]

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

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

print(f"\n[{time.time() - total_start:.2f}s] Evaluation complete")
# %%
