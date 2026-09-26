from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "combined" / "processed"

IMAGE_SIZE = 28

# Numero di immagini visualizzate
NUM_IMAGES = 50

# Numero di immagini per riga
COLUMNS = 10

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

    Le immagini vengono selezionate casualmente ad ogni
    esecuzione del programma.
    """

    num_images = min(
        NUM_IMAGES,
        len(X),
    )

    # Nessun seed:
    # ogni esecuzione produce un campione diverso.
    rng = np.random.default_rng()

    indices = rng.choice(
        len(X),
        size=num_images,
        replace=False,
    )

    rows = int(
        np.ceil(
            num_images / COLUMNS
        )
    )

    plt.figure(
        figsize=(16, 2 * rows)
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
            COLUMNS,
            position + 1,
        )

        plt.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        plt.title(
            letter,
            fontsize=10,
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