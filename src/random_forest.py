# %%
import time
import numpy as np
import seaborn as sns
import sklearn.metrics
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt


# %%
def load_dataset(train: bool):
    print(f"Loading {'train' if train else 'test'} dataset...")
    dataset_train = torchvision.datasets.FashionMNIST(
        "./data",
        train=train,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    loader = torch.utils.data.DataLoader(dataset_train, batch_size=1024, shuffle=False)

    Xs = []
    ys = []
    for imgs, labels in loader:
        Xs.append(imgs.view(imgs.size(0), -1).numpy())
        ys.append(labels.numpy())

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


print("=" * 60)
print("FASHION-MNIST RANDOM FOREST CLASSIFIER")
print("=" * 60)


print("\n[DATA LOADING]")
X_train, y_train = load_dataset(train=True)
X_test, y_test = load_dataset(train=False)
print(
    f"Data loading complete: {X_train.shape[0]} train, {X_test.shape[0]} test samples"
)

print("\n[MODEL TRAINING]")
clf = RandomForestClassifier(
    n_estimators=100, criterion="entropy", max_depth=100, n_jobs=-1
)
print("Random Forest initialized: n_estimators=100, max_depth=100, criterion=entropy")

train_start = time.time()
clf.fit(X_train, y_train)
train_time = time.time() - train_start
print(f"Training completed in {train_time:.2f} seconds")

print("\n[MODEL EVALUATION]")
eval_start = time.time()
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
eval_time = time.time() - eval_start
print(f"Evaluation completed in {eval_time:.2f} seconds")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Train accuracy:  {clf.score(X_train, y_train):.4f}")
print(f"Test accuracy:   {clf.score(X_test, y_test):.4f}")
print(
    f"Test precision:  {sklearn.metrics.precision_score(y_test, pred_test, average='weighted'):.4f}"
)
print(
    f"Test recall:     {sklearn.metrics.recall_score(y_test, pred_test, average='weighted'):.4f}"
)
print("=" * 60)


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
# %%
