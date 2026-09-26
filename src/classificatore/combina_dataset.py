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

DIGITAL_TEST_SIZE = 0.15

CLASS_NAMES = {
    0: "C",
    1: "D",
    2: "F",
    3: "S",
    4: "V",
    5: "X",
}


# ============================================================
# CARICAMENTO DATASET DIGITALE
# ============================================================

def load_digital():
    X = np.load(DIGITAL_DIR / "X.npy")
    y = np.load(DIGITAL_DIR / "y.npy")

    print("DIGITAL:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print()

    return X, y


# ============================================================
# CARICAMENTO EMNIST
# ============================================================

def load_emnist():

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
    X,
    y,
    name,
):

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
    labels,
    name,
):

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
# COMBINAZIONE DATASET
# ============================================================

def combine_datasets():

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

    X_digital, y_digital = load_digital()

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
    # DIVISIONE DIGITAL TRAIN / TEST
    # --------------------------------------------------------

    print("Divisione DIGITAL train/test...")

    (
        X_digital_train,
        X_digital_test,
        y_digital_train,
        y_digital_test,
    ) = train_test_split(
        X_digital,
        y_digital,
        test_size=DIGITAL_TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_digital,
    )

    print(
        f"DIGITAL train: {X_digital_train.shape}"
    )

    print(
        f"DIGITAL test:  {X_digital_test.shape}"
    )

    print()

    # --------------------------------------------------------
    # UNIONE EMNIST TRAIN + DIGITAL TRAIN
    # --------------------------------------------------------

    print(
        "Unione EMNIST train + DIGITAL train..."
    )

    X_train = np.concatenate(
        [
            X_emnist_train,
            X_digital_train,
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            y_emnist_train,
            y_digital_train,
        ],
        axis=0,
    )

    print(
        f"Dataset TRAIN: {X_train.shape}"
    )

    print()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    X_val = X_emnist_val
    y_val = y_emnist_val

    # --------------------------------------------------------
    # TEST EMNIST
    # --------------------------------------------------------

    X_test_emnist = X_emnist_test
    y_test_emnist = y_emnist_test

    # --------------------------------------------------------
    # SHUFFLE TRAIN
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
    # DISTRIBUZIONE CLASSI
    # --------------------------------------------------------

    print_class_distribution(
        y_train,
        "TRAIN",
    )

    print_class_distribution(
        y_val,
        "VALIDATION",
    )

    print_class_distribution(
        y_test_emnist,
        "TEST EMNIST",
    )

    print_class_distribution(
        y_digital_test,
        "TEST DIGITAL",
    )

    # --------------------------------------------------------
    # CREAZIONE CARTELLA
    # --------------------------------------------------------

    COMBINED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # SALVATAGGIO TRAIN
    # --------------------------------------------------------

    np.save(
        COMBINED_DIR / "X_train.npy",
        X_train,
    )

    np.save(
        COMBINED_DIR / "y_train.npy",
        y_train,
    )

    # --------------------------------------------------------
    # SALVATAGGIO VALIDATION
    # --------------------------------------------------------

    np.save(
        COMBINED_DIR / "X_val.npy",
        X_val,
    )

    np.save(
        COMBINED_DIR / "y_val.npy",
        y_val,
    )

    # --------------------------------------------------------
    # SALVATAGGIO TEST EMNIST
    # --------------------------------------------------------

    np.save(
        COMBINED_DIR / "X_test_emnist.npy",
        X_test_emnist,
    )

    np.save(
        COMBINED_DIR / "y_test_emnist.npy",
        y_test_emnist,
    )

    # --------------------------------------------------------
    # SALVATAGGIO TEST DIGITAL
    # --------------------------------------------------------

    np.save(
        COMBINED_DIR / "X_test_digital.npy",
        X_digital_test,
    )

    np.save(
        COMBINED_DIR / "y_test_digital.npy",
        y_digital_test,
    )

    # --------------------------------------------------------
    # RISULTATO
    # --------------------------------------------------------

    print("========================================")
    print("DATASET COMBINATO CREATO")
    print("========================================")
    print()

    print(f"X_train:        {X_train.shape}")
    print(f"y_train:        {y_train.shape}")

    print(f"X_val:          {X_val.shape}")
    print(f"y_val:          {y_val.shape}")

    print(
        f"X_test_emnist:  {X_test_emnist.shape}"
    )

    print(
        f"y_test_emnist:  {y_test_emnist.shape}"
    )

    print(
        f"X_test_digital: {X_digital_test.shape}"
    )

    print(
        f"y_test_digital: {y_digital_test.shape}"
    )

    print()
    print("Salvato in:")
    print(COMBINED_DIR)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    combine_datasets()