from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "emnist" / "processed"


CLASS_NAMES = {
    0: "C",
    1: "D",
    2: "F",
    3: "S",
    4: "V",
    5: "X",
}


def main():
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")

    print(f"Dataset caricato: {X_train.shape}")

    fig, axes = plt.subplots(6, 5, figsize=(10, 12))

    for class_id in range(6):
        indices = np.where(y_train == class_id)[0][:5]

        for column, index in enumerate(indices):
            image = X_train[index].reshape(28, 28)

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