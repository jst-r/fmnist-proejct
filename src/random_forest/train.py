"""Training script for Random Forest classifier."""

import time

from src.random_forest.shared import load_dataset, make_model, save_model


def main():
    print("=" * 60)
    print("FASHION-MNIST RANDOM FOREST CLASSIFIER - TRAINING")
    print("=" * 60)

    print("\n[DATA LOADING]")
    X_train, y_train = load_dataset(train=True)
    X_test, y_test = load_dataset(train=False)
    print(
        f"Data loading complete: {X_train.shape[0]} train, {X_test.shape[0]} test samples"
    )

    print("\n[MODEL TRAINING]")
    clf, save_path = make_model(
        n_estimators=100, criterion="entropy", max_depth=100, n_jobs=-1
    )
    print(
        "Random Forest initialized: n_estimators=100, max_depth=100, criterion=entropy"
    )

    train_start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.2f} seconds")

    print("\n[INITIAL EVALUATION]")
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    print("\n[MODEL SAVING]")
    save_model(clf, save_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
