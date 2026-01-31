import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn.metrics


def print_classification_metrics(y_true, y_pred, class_names):
    """Print overall and per-class classification metrics."""
    print("=" * 60)
    print(f"Test accuracy:   {sklearn.metrics.accuracy_score(y_true, y_pred):.4f}")
    print(
        f"Test precision:  {sklearn.metrics.precision_score(y_true, y_pred, average='weighted'):.4f}"
    )
    print(
        f"Test recall:     {sklearn.metrics.recall_score(y_true, y_pred, average='weighted'):.4f}"
    )
    print("=" * 60)

    for label in range(len(class_names)):
        print(f"Label: {class_names[label]}")
        print(
            "Precision:\t{:.4f}".format(
                sklearn.metrics.precision_score(
                    y_true, y_pred, average="weighted", labels=[label]
                ),
            )
        )
        print(
            "Recall:\t\t{:.4f}".format(
                sklearn.metrics.recall_score(
                    y_true, y_pred, average="weighted", labels=[label]
                ),
            )
        )


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot and save confusion matrix heatmap."""
    confusion = sklearn.metrics.confusion_matrix(y_true, y_pred)
    norm_confusion = np.round(confusion / confusion.sum(axis=1, keepdims=True), 2)
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
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_misclassified(
    x_data,
    y_true,
    y_pred,
    class_names,
    output_path,
    n_pairs=9,
    samples_per_pair=9,
    img_size_inches=1,
):
    """Plot and save misclassified examples grouped by error type."""
    # Find incorrect predictions
    incorrect_mask = y_pred != y_true
    total_incorrect = incorrect_mask.sum()
    incorrect_indices = np.where(incorrect_mask)[0]

    if total_incorrect == 0:
        print("No misclassifications to plot!")
        return

    # Group by (true_label, pred_label) pairs
    pair_counts = {}
    pair_indices = {}

    for idx in incorrect_indices:
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        pair = (true_label, pred_label)

        if pair not in pair_counts:
            pair_counts[pair] = 0
            pair_indices[pair] = []
        pair_counts[pair] += 1
        pair_indices[pair].append(idx)

    # Get top N most common misclassification pairs
    top_n_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[
        :n_pairs
    ]

    print(f"Top {n_pairs} misclassification pairs:")
    shown_incorrect = 0
    for (true_label, pred_label), count in top_n_pairs:
        shown_incorrect += count
        print(
            f"  {class_names[true_label]} -> {class_names[pred_label]}: {count} examples"
        )

    print(
        f"Sampling {shown_incorrect} examples from {total_incorrect} total. {shown_incorrect / total_incorrect:.2%} of total"
    )

    # Create figure with n_pairs rows x (1 label column + samples_per_pair image columns)
    fig_width = img_size_inches * (samples_per_pair + 2)
    fig_height = img_size_inches * n_pairs
    fig, axes = plt.subplots(
        n_pairs, samples_per_pair + 1, figsize=(fig_width, fig_height)
    )
    fig.suptitle(
        f"Top {n_pairs} Misclassification Pairs ({samples_per_pair} Random Samples Each)",
        fontsize=14,
        y=0.98,
    )

    for row, ((true_label, pred_label), count) in enumerate(top_n_pairs):
        indices = pair_indices[(true_label, pred_label)]

        # Randomly sample examples (or fewer if not enough)
        sample_size = min(samples_per_pair, len(indices))
        sampled_indices = np.random.choice(indices, size=sample_size, replace=False)

        # First column: label
        label_ax = axes[row, 0]
        label_ax.text(
            0.5,
            0.5,
            f"True:\n{class_names[true_label]}\nPrediction:\n{class_names[pred_label]}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            transform=label_ax.transAxes,
        )
        label_ax.axis("off")

        # Remaining columns: images
        for col in range(samples_per_pair):
            ax = axes[row, col + 1]

            if col < sample_size:
                idx = sampled_indices[col]
                img = x_data[idx]

                # Convert from (C, H, W) to (H, W, C) for display
                if img.shape[0] == 1:  # Grayscale
                    img = img.squeeze(0)
                else:
                    img = np.transpose(img, (1, 2, 0))

                ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
            ax.axis("off")

    # Set fixed 2-pixel margins between images
    # wspace/hspace are in fraction of subplot size; 2px at 100 DPI = 0.02 inches
    # With img_size_inches=1.5, spacing = 0.02/1.5 ≈ 0.013
    spacing = 2 / (plt.gcf().dpi * img_size_inches)
    plt.subplots_adjust(wspace=spacing, hspace=spacing, top=0.95)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
