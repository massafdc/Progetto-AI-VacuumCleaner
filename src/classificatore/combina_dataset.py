from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EMNIST_DIR = (
    BASE_DIR
    / "data"
    / "emnist"
    / "processed"
)

DIGITAL_DIR = (
    BASE_DIR
    / "data"
    / "digital"
    / "processed"
)

COMBINED_DIR = (
    BASE_DIR
    / "data"
    / "combined"
    / "processed"
)


# ============================================================
# CONFIGURAZIONE
# ============================================================

RANDOM_SEED = 42

VALIDATION_SIZE = 0.15

CLASS_NAMES = {
    0: "C",
    1: "D",
    2: "F",
    3: "S",
    4: "V",
    5: "X",
}


# ============================================================
# CARICAMENTO
# ============================================================

def load_dataset(
    directory: Path,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:

    X = np.load(directory / "X.npy")
    y = np.load(directory / "y.npy")

    print(f"{name}:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")

    return X, y


def load_emnist() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    X_train = np.load(
        EMNIST_DIR / "X_train.npy"
    )

    y_train = np.load(
        EMNIST_DIR / "y_train.npy"
    )

    X_val = np.load(
        EMNIST_DIR / "X_val.npy"
    )

    y_val = np.load(
        EMNIST_DIR / "y_val.npy"
    )

    X_test = np.load(
        EMNIST_DIR / "X_test.npy"
    )

    y_test = np.load(
        EMNIST_DIR / "y_test.npy"
    )

    print("EMNIST:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")
    print()

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# CONTROLLO FORMATO
# ============================================================

def validate_dataset(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
) -> None:

    if X.ndim != 2:
        raise ValueError(
            f"{name}: X deve avere forma (N, 784), "
            f"ma ha forma {X.shape}"
        )

    if X.shape[1] != 784:
        raise ValueError(
            f"{name}: X deve contenere 784 pixel, "
            f"ma ne contiene {X.shape[1]}"
        )

    if y.ndim != 1:
        raise ValueError(
            f"{name}: y deve avere forma (N,), "
            f"ma ha forma {y.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"{name}: numero di immagini e label diverso."
        )


# ============================================================
# DISTRIBUZIONE CLASSI
# ============================================================

def print_class_distribution(
    labels: np.ndarray,
    name: str,
) -> None:

    print(f"Distribuzione {name}:")

    for class_id, class_name in CLASS_NAMES.items():

        count = np.sum(
            labels == class_id
        )

        print(
            f"  {class_id} ({class_name}): "
            f"{count}"
        )

    print()


# ============================================================
# MAIN
# ============================================================

def combine_datasets() -> None:

    print("========================================")
    print("UNIONE DATASET")
    print("========================================")
    print()

    # --------------------------------------------------------
    # CARICAMENTO EMNIST
    # --------------------------------------------------------

    (
        X_emnist_train,
        y_emnist_train,
        X_emnist_val,
        y_emnist_val,
        X_emnist_test,
        y_emnist_test,
    ) = load_emnist()

    # --------------------------------------------------------
    # CARICAMENTO DIGITALE
    # --------------------------------------------------------

    X_digital, y_digital = load_dataset(
        DIGITAL_DIR,
        "DIGITAL",
    )

    print()

    # --------------------------------------------------------
    # CONTROLLO FORMATI
    # --------------------------------------------------------

    validate_dataset(
        X_emnist_train,
        y_emnist_train,
        "EMNIST train",
    )

    validate_dataset(
        X_emnist_val,
        y_emnist_val,
        "EMNIST validation",
    )

    validate_dataset(
        X_emnist_test,
        y_emnist_test,
        "EMNIST test",
    )

    validate_dataset(
        X_digital,
        y_digital,
        "DIGITAL",
    )

    # --------------------------------------------------------
    # UNIONE TRAIN
    # --------------------------------------------------------

    print("Unione EMNIST train + DIGITAL...")

    X_train_full = np.concatenate(
        [
            X_emnist_train,
            X_digital,
        ],
        axis=0,
    )

    y_train_full = np.concatenate(
        [
            y_emnist_train,
            y_digital,
        ],
        axis=0,
    )

    print(
        f"Dataset unito: {X_train_full.shape}"
    )

    print()

    # --------------------------------------------------------
    # NUOVA DIVISIONE TRAIN / VALIDATION
    # --------------------------------------------------------

    print("Creazione train/validation...")

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_SEED,
            stratify=y_train_full,
        )
    )

    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    permutation = rng.permutation(
        len(X_train)
    )

    X_train = X_train[permutation]
    y_train = y_train[permutation]

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------
    #
    # Manteniamo il test EMNIST separato.
    #
    # Questo permette di misurare le prestazioni
    # del classificatore su dati EMNIST mai utilizzati
    # durante il training.
    #
    # --------------------------------------------------------

    X_test = X_emnist_test
    y_test = y_emnist_test

    # --------------------------------------------------------
    # DISTRIBUZIONE
    # --------------------------------------------------------

    print()
    print_class_distribution(
        y_train,
        "TRAIN",
    )

    print_class_distribution(
        y_val,
        "VALIDATION",
    )

    print_class_distribution(
        y_test,
        "TEST EMNIST",
    )

    # --------------------------------------------------------
    # SALVATAGGIO
    # --------------------------------------------------------

    COMBINED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        COMBINED_DIR / "X_train.npy",
        X_train,
    )

    np.save(
        COMBINED_DIR / "y_train.npy",
        y_train,
    )

    np.save(
        COMBINED_DIR / "X_val.npy",
        X_val,
    )

    np.save(
        COMBINED_DIR / "y_val.npy",
        y_val,
    )

    np.save(
        COMBINED_DIR / "X_test.npy",
        X_test,
    )

    np.save(
        COMBINED_DIR / "y_test.npy",
        y_test,
    )

    # --------------------------------------------------------
    # RISULTATO
    # --------------------------------------------------------

    print("========================================")
    print("DATASET COMBINATO CREATO")
    print("========================================")
    print()

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    print(f"X_val:   {X_val.shape}")
    print(f"y_val:   {y_val.shape}")

    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")

    print()
    print(f"Salvato in:")
    print(COMBINED_DIR)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    combine_datasets()