"""Generate dataset sample visualization for Fashion-MNIST."""

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

from src.common.paths import DATA_DIR, PROJECT_DIR
from src.common.shared import CLASS_NAMES


def get_first_n_samples_per_class(dataset, n_samples=5):
    """Get first n samples from each class."""
    samples_by_class = {i: [] for i in range(10)}

    for idx, (img, label) in enumerate(dataset):
        if len(samples_by_class[label]) < n_samples:
            samples_by_class[label].append(img)

        # Check if we have enough samples for all classes
        if all(len(samples) >= n_samples for samples in samples_by_class.values()):
            break

    return samples_by_class


def plot_dataset_samples(output_path, n_samples=5):
    """Create a grid showing n samples from each Fashion-MNIST class."""
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.FashionMNIST(
        DATA_DIR, train=True, download=True, transform=transform
    )

    # Get samples
    samples_by_class = get_first_n_samples_per_class(dataset, n_samples)

    # Create figure: 10 rows (classes) × n_samples columns
    fig, axes = plt.subplots(10, n_samples, figsize=(n_samples * 1.2, 10 * 1.2))
    fig.suptitle(
        "Fashion-MNIST Dataset Samples (First 5 per Class)", fontsize=14, y=0.995
    )

    for class_idx in range(10):
        for sample_idx in range(n_samples):
            ax = axes[class_idx, sample_idx]
            img = samples_by_class[class_idx][sample_idx]

            # Convert tensor to numpy and squeeze channel dimension
            img_np = img.squeeze().numpy()

            ax.imshow(img_np, cmap="gray")
            ax.axis("off")

            # Add class label only on first column
            if sample_idx == 0:
                ax.text(
                    -0.5,
                    0.5,
                    CLASS_NAMES[class_idx],
                    ha="right",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    transform=ax.transAxes,
                )

    plt.tight_layout()
    plt.subplots_adjust(left=0.15, top=0.98)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Dataset samples visualization saved to {output_path}")


if __name__ == "__main__":
    output_dir = PROJECT_DIR / "plots" / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "samples.png"
    plot_dataset_samples(output_path, n_samples=5)
