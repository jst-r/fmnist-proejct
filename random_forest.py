# %%
import numpy as np
import seaborn as sns
import sklearn.metrics
import torch
import torchvision
from sklearn.ensemble import RandomForestClassifier


# %%
def load_dataset(train: bool):
    dataset_train = torchvision.datasets.FashionMNIST(
        "./dataset",
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
pred = clf.predict(X_test)
print("Train accuracy:", clf.score(X_train, y_train))

print("Test accuracy:", clf.score(X_test, y_test))

print(
    "Test precision:", sklearn.metrics.precision_score(y_test, pred, average="weighted")
)
print("Test recall:", sklearn.metrics.recall_score(y_test, pred, average="weighted"))

for label in range(10):
    print(
        "Test precision for label {}: {}".format(
            label,
            sklearn.metrics.precision_score(
                y_test, pred, average="weighted", labels=[label]
            ),
        )
    )
    print(
        "Test recall for label {}: {}".format(
            label,
            sklearn.metrics.recall_score(
                y_test, pred, average="weighted", labels=[label]
            ),
        )
    )

# %%
sns.heatmap(sklearn.metrics.confusion_matrix(y_test, pred))

# %%
