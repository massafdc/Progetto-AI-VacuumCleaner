from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "combined"
    / "processed"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_PATH = MODEL_DIR / "letter_classifier.joblib"


# ============================================================
# CLASSI
# ============================================================

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
    Carica il dataset combinato.
    """

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    X_val = np.load(
        DATA_DIR / "X_val.npy"
    )

    y_val = np.load(
        DATA_DIR / "y_val.npy"
    )

    X_test_emnist = np.load(
        DATA_DIR / "X_test_emnist.npy"
    )

    y_test_emnist = np.load(
        DATA_DIR / "y_test_emnist.npy"
    )

    X_test_digital = np.load(
        DATA_DIR / "X_test_digital.npy"
    )

    y_test_digital = np.load(
        DATA_DIR / "y_test_digital.npy"
    )

    print("Dataset caricato:")

    print(
        f"  X_train:        {X_train.shape}"
    )

    print(
        f"  y_train:        {y_train.shape}"
    )

    print(
        f"  X_val:          {X_val.shape}"
    )

    print(
        f"  y_val:          {y_val.shape}"
    )

    print(
        f"  X_test_emnist:  {X_test_emnist.shape}"
    )

    print(
        f"  y_test_emnist:  {y_test_emnist.shape}"
    )

    print(
        f"  X_test_digital: {X_test_digital.shape}"
    )

    print(
        f"  y_test_digital: {y_test_digital.shape}"
    )

    print()

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test_emnist,
        y_test_emnist,
        X_test_digital,
        y_test_digital,
    )


# ============================================================
# MODELLO
# ============================================================

def create_model() -> MLPClassifier:
    """
    Crea il classificatore MLP.
    """

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=30,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        verbose=True,
    )

    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model: MLPClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    """
    Addestra il modello.
    """

    print("Inizio training...\n")

    model.fit(
        X_train,
        y_train,
    )

    print("\nTraining completato.")

    return model


# ============================================================
# VALUTAZIONE
# ============================================================

def evaluate_model(
    model: MLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
    name: str,
):
    """
    Valuta il classificatore.
    """

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    print()
    print("=" * 60)
    print(f"RISULTATI {name}")
    print("=" * 60)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                CLASS_NAMES[i]
                for i in range(len(CLASS_NAMES))
            ],
            digits=4,
        )
    )

    return accuracy


# ============================================================
# SALVATAGGIO
# ============================================================

def save_model(
    model: MLPClassifier,
):
    """
    Salva il modello addestrato.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"Modello salvato in:\n{MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test_emnist,
        y_test_emnist,
        X_test_digital,
        y_test_digital,
    ) = load_dataset()

    # --------------------------------------------------------
    # MODELLO
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    train_model(
        model,
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_val,
        y_val,
        "VALIDATION",
    )

    # --------------------------------------------------------
    # TEST EMNIST
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_test_emnist,
        y_test_emnist,
        "TEST EMNIST",
    )

    # --------------------------------------------------------
    # TEST DIGITAL
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_test_digital,
        y_test_digital,
        "TEST DIGITAL",
    )

    # --------------------------------------------------------
    # SALVATAGGIO
    # --------------------------------------------------------

    save_model(model)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()