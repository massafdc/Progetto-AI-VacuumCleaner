from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "digital" / "processed"


CLASS_NAMES = {
    0: "C",
    1: "D",
    2: "F",
    3: "S",
    4: "V",
    5: "X",
}


def main():
    X = np.load(PROCESSED_DIR / "X.npy")
    y = np.load(PROCESSED_DIR / "y.npy")

    print(f"Dataset caricato: {X.shape}")
    print(f"Label caricate:   {y.shape}")

    fig, axes = plt.subplots(6, 5, figsize=(10, 12))

    for class_id in range(6):
        indices = np.where(y == class_id)[0][:5]

        for column, index in enumerate(indices):
            image = X[index].reshape(28, 28)

            axes[class_id, column].imshow(
                image,
                cmap="gray"
            )

            axes[class_id, column].set_title(
                CLASS_NAMES[class_id]
            )

            axes[class_id, column].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()