
from pathlib import Path
import gzip
import struct

import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "emnist" / "raw" / "gzip"
PROCESSED_DIR = BASE_DIR / "data" / "emnist" / "processed"


# ============================================================
# CLASSI CHE CI INTERESSANO
# ============================================================

# Label originali EMNIST ByClass:
#
# 12 -> C
# 13 -> D
# 15 -> F
# 28 -> S
# 21 -> V
# 33 -> X

TARGET_LABELS = {
    12: 0,  # C
    13: 1,  # D
    15: 2,  # F
    28: 3,  # S
    21: 4,  # V
    33: 5,  # X
}

CLASS_NAMES = {
    0: "C",
    1: "D",
    2: "F",
    3: "S",
    4: "V",
    5: "X",
}


# ============================================================
# LETTURA FILE IDX
# ============================================================

def read_idx_images(path: Path) -> np.ndarray:
    """
    Legge un file IDX contenente immagini EMNIST.

    Restituisce un array di forma:
        (numero_immagini, 28, 28)
    """

    open_func = gzip.open if path.suffix == ".gz" else open

    with open_func(path, "rb") as file:
        magic, num_images, rows, cols = struct.unpack(
            ">IIII",
            file.read(16)
        )

        if magic != 2051:
            raise ValueError(
                f"Magic number non valido per immagini: {magic}"
            )

        data = np.frombuffer(
            file.read(),
            dtype=np.uint8
        )

    expected_size = num_images * rows * cols

    if data.size != expected_size:
        raise ValueError(
            f"Dimensione dati non valida in {path.name}: "
            f"atteso {expected_size}, trovato {data.size}"
        )

    return data.reshape(num_images, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    """
    Legge un file IDX contenente le label EMNIST.

    Restituisce un array di forma:
        (numero_immagini,)
    """

    open_func = gzip.open if path.suffix == ".gz" else open

    with open_func(path, "rb") as file:
        magic, num_labels = struct.unpack(
            ">II",
            file.read(8)
        )

        if magic != 2049:
            raise ValueError(
                f"Magic number non valido per label: {magic}"
            )

        labels = np.frombuffer(
            file.read(),
            dtype=np.uint8
        )

    if labels.size != num_labels:
        raise ValueError(
            f"Numero di label non valido in {path.name}: "
            f"atteso {num_labels}, trovato {labels.size}"
        )

    return labels


# ============================================================
# PREPROCESSING IMMAGINI
# ============================================================

def preprocess_images(images: np.ndarray) -> np.ndarray:
    """
    Corregge l'orientamento delle immagini EMNIST,
    normalizza i pixel e appiattisce ogni immagine.

    Input:
        (N, 28, 28)

    Output:
        (N, 784)
    """

    # EMNIST memorizza le immagini trasposte rispetto
    # all'orientamento normalmente utilizzato per la visualizzazione.
    #
    # Trasponiamo ogni immagine per correggerne l'orientamento.
    images = np.transpose(images, (0, 2, 1))

    # Conversione a float32 e normalizzazione dei pixel
    # da [0, 255] a [0, 1].
    images = images.astype(np.float32) / 255.0

    # Appiattiamo ogni immagine 28x28 in un vettore di 784 valori.
    images = images.reshape(images.shape[0], -1)

    return images


# ============================================================
# SELEZIONE DELLE CLASSI
# ============================================================

def select_target_classes(
    images: np.ndarray,
    labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    mask = np.isin(labels, list(TARGET_LABELS.keys()))

    selected_images = images[mask]
    selected_labels = labels[mask]

    # Remapping delle label EMNIST:
    #
    # 12 -> 0 -> C
    # 13 -> 1 -> D
    # 15 -> 2 -> F
    # 28 -> 3 -> S
    # 21 -> 4 -> V
    # 33 -> 5 -> X

    remapped_labels = np.array(
        [TARGET_LABELS[label] for label in selected_labels],
        dtype=np.int64
    )

    return selected_images, remapped_labels


# ============================================================
# STAMPA DISTRIBUZIONE CLASSI
# ============================================================

def print_class_distribution(labels: np.ndarray, name: str) -> None:
    print(f"\nDistribuzione {name}:")

    for class_id, class_name in CLASS_NAMES.items():
        count = np.sum(labels == class_id)
        print(f"  {class_id} ({class_name}): {count}")


# ============================================================
# MAIN
# ============================================================

def prepare_dataset() -> None:

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Caricamento EMNIST ByClass...\n")

    train_images_path = (
        RAW_DIR / "emnist-byclass-train-images-idx3-ubyte.gz"
    )

    train_labels_path = (
        RAW_DIR / "emnist-byclass-train-labels-idx1-ubyte.gz"
    )

    test_images_path = (
        RAW_DIR / "emnist-byclass-test-images-idx3-ubyte.gz"
    )

    test_labels_path = (
        RAW_DIR / "emnist-byclass-test-labels-idx1-ubyte.gz"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("Lettura training set...")

    train_images = read_idx_images(train_images_path)
    train_labels = read_idx_labels(train_labels_path)

    print(f"Immagini train originali: {train_images.shape}")
    print(f"Label train originali:    {train_labels.shape}")

    train_images, train_labels = select_target_classes(
        train_images,
        train_labels
    )

    print(
        f"\nDopo selezione classi: {train_images.shape}"
    )

    print_class_distribution(
        train_labels,
        "training selezionato"
    )

    train_images = preprocess_images(train_images)

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\nLettura test set...")

    test_images = read_idx_images(test_images_path)
    test_labels = read_idx_labels(test_labels_path)

    print(f"Immagini test originali: {test_images.shape}")
    print(f"Label test originali:    {test_labels.shape}")

    test_images, test_labels = select_target_classes(
        test_images,
        test_labels
    )

    print(
        f"\nDopo selezione classi: {test_images.shape}"
    )

    print_class_distribution(
        test_labels,
        "test selezionato"
    )

    test_images = preprocess_images(test_images)

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        train_images,
        train_labels,
        test_size=0.15,
        random_state=42,
        stratify=train_labels
    )

    print("\nDimensioni finali:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"y_val:   {y_val.shape}")
    print(f"X_test:  {test_images.shape}")
    print(f"y_test:  {test_labels.shape}")

    # --------------------------------------------------------
    # SALVATAGGIO
    # --------------------------------------------------------

    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)

    np.save(PROCESSED_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)

    np.save(PROCESSED_DIR / "X_test.npy", test_images)
    np.save(PROCESSED_DIR / "y_test.npy", test_labels)

    print("\nDataset preprocessato salvato in:")
    print(PROCESSED_DIR)


if __name__ == "__main__":
    prepare_dataset()

