# %%
import time
import numpy as np

from src.random_forest.shared import load_dataset, load_model, MODEL_PATH
from src.common.eval import (
    print_classification_metrics,
    plot_confusion_matrix,
    plot_misclassified,
)
from src.common.paths import PROJECT_DIR
from src.common.shared import CLASS_NAMES


print("=" * 60)
print("FASHION-MNIST RANDOM FOREST EVALUATION")
print("=" * 60)

total_start = time.time()

# %%
print("\n[MODEL LOADING]")
clf = load_model(MODEL_PATH)
print(f"[{time.time() - total_start:.2f}s] Model loaded from {MODEL_PATH}")

# %%
print("\n[DATA LOADING]")
data_start = time.time()
X_train, y_train = load_dataset(train=True)
X_test, y_test = load_dataset(train=False)
print(
    f"[{time.time() - total_start:.2f}s] Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples"
)

# %%
print("\n[PREDICTION]")
pred_start = time.time()


def get_predictions(X, y, dataset_name):
    pred_start = time.time()
    preds = clf.predict(X)
    elapsed = time.time() - pred_start
    print(
        f"[{time.time() - total_start:.2f}s] Generated predictions for {dataset_name} set in {elapsed:.2f}s"
    )
    return preds, y, X


# Get predictions for test set
pred_test, y_test, X_test = get_predictions(X_test, y_test, "test")

# %%
print("\n[RESULTS]")
print_classification_metrics(y_test, pred_test, CLASS_NAMES)

# %% Plot confusion matrix
plot_confusion_matrix(
    y_test, pred_test, CLASS_NAMES, PROJECT_DIR / "plots/random_forest/confusion.png"
)
print(f"[{time.time() - total_start:.2f}s] Confusion matrix saved")

# %% Visualize misclassified examples
print("\n[MISCLASSIFIED EXAMPLES]")


# Reshape X_test back to image format for plotting
def reshape_for_plotting(X):
    """Reshape flattened images back to (N, C, H, W) format."""
    n_samples = X.shape[0]
    return X.reshape(n_samples, 1, 28, 28)


X_test_images = reshape_for_plotting(X_test)
plot_misclassified(
    X_test_images,
    y_test,
    pred_test,
    CLASS_NAMES,
    PROJECT_DIR / "plots/random_forest/misclassified.png",
)
print(f"[{time.time() - total_start:.2f}s] Misclassified examples saved")

print(f"\n[{time.time() - total_start:.2f}s] Evaluation complete")
# %%
