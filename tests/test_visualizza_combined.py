from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "combined" / "processed"

IMAGE_SIZE = 28

NUM_IMAGES = 25

RANDOM_SEED = 42

CLASS_NAMES = {
    0: "C",
    1: "D",
    2: "F",
    3: "S",
    4: "V",
    5: "X",
}


# ============================================================
# CARICAMENTO DATASET
# ============================================================

def load_dataset():
    """
    Carica il training set del dataset combinato.
    """

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    print("Dataset combinato caricato:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print()

    return X_train, y_train


# ============================================================
# VISUALIZZAZIONE
# ============================================================

def show_images(X, y):
    """
    Visualizza un campione casuale di immagini.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    num_images = min(
        NUM_IMAGES,
        len(X),
    )

    indices = rng.choice(
        len(X),
        size=num_images,
        replace=False,
    )

    columns = 5
    rows = int(np.ceil(num_images / columns))

    plt.figure(
        figsize=(10, 10)
    )

    for position, index in enumerate(indices):

        image = X[index].reshape(
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        label = y[index]

        letter = CLASS_NAMES[int(label)]

        plt.subplot(
            rows,
            columns,
            position + 1,
        )

        plt.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        plt.title(
            f"Classe: {letter}"
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VISUALIZZAZIONE DATASET COMBINATO")
    print("=" * 60)
    print()

    X_train, y_train = load_dataset()

    show_images(
        X_train,
        y_train,
    )


if __name__ == "__main__":
    main()