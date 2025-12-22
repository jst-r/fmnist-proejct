# %%
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn.metrics
import torch

from shared import get_loaders, make_model, device

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
# %% Load weights
model, save_path = make_model()
model.to(device)
# Ensure the path matches your saved file
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()


# %% Dataloader (Keep images as tensors for the CNN)
def get_predictions(train: bool):
    train_loader, test_loader = get_loaders(batch_size=512, use_workers=False)
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

    return np.concatenate(all_preds), np.concatenate(all_labels)


# %% Get Results
pred_train, y_train = get_predictions(train=True)
pred_test, y_test = get_predictions(train=False)

# %% Metrics (Same as your RF script)
print(
    "Train accuracy:\t{:.4f}".format(sklearn.metrics.accuracy_score(y_train, pred_train))
)
print("Test accuracy:\t{:.4f}".format(sklearn.metrics.accuracy_score(y_test, pred_test)))

print(
    "Test precision:\t{:.4f}".format(
        sklearn.metrics.precision_score(y_test, pred_test, average="weighted")
    ),
)
print(
    "Test recall:\t{:.4f}".format(
        sklearn.metrics.recall_score(y_test, pred_test, average="weighted")
    )
)

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
log_confusion = np.log1p(confusion)
sns.heatmap(
    log_confusion,
    annot=confusion,
    fmt="d",
    cbar=False,
    square=True,
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
# %%
