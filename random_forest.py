# %%
import numpy as np
import seaborn as sns
import sklearn.metrics
import torch
import torchvision
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt


# %%
def load_dataset(train: bool):
    dataset_train = torchvision.datasets.FashionMNIST(
        "./data",
        train=train,
        download=True,
        transform=torchvision.transforms.ToTensor(),
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


X_train, y_train = load_dataset(train=True)
X_test, y_test = load_dataset(train=False)
# %%
clf = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=100)

# %%
clf.fit(X_train, y_train)
# %%
pred_test = clf.predict(X_test)
print("Train accuracy:", clf.score(X_train, y_train))

print("Test accuracy:", clf.score(X_test, y_test))

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
