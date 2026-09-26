from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


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

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "letter_classifier.joblib"
)


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

def load_test_dataset(dataset_name):
    """
    Carica uno dei due test set:
    - EMNIST
    - DIGITAL
    """

    if dataset_name == "EMNIST":

        X_test = np.load(
            DATA_DIR / "X_test_emnist.npy"
        )

        y_test = np.load(
            DATA_DIR / "y_test_emnist.npy"
        )

    elif dataset_name == "DIGITAL":

        X_test = np.load(
            DATA_DIR / "X_test_digital.npy"
        )

        y_test = np.load(
            DATA_DIR / "y_test_digital.npy"
        )

    else:

        raise ValueError(
            f"Dataset non riconosciuto: {dataset_name}"
        )

    print(f"Test dataset {dataset_name} caricato:")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    print()

    return X_test, y_test


# ============================================================
# CARICAMENTO MODELLO
# ============================================================

def load_model():
    """
    Carica il classificatore addestrato.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modello non trovato:\n{MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print("Modello caricato:")
    print(MODEL_PATH)
    print()

    return model


# ============================================================
# PREDIZIONI
# ============================================================

def make_predictions(
    model,
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Genera le predizioni sul test set.
    """

    print("Generazione predizioni...")

    predictions = model.predict(
        X_test
    )

    print("Predizioni completate.")
    print()

    return predictions


# ============================================================
# ACCURACY
# ============================================================

def evaluate_accuracy(
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """
    Calcola e stampa l'accuracy.
    """

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("=" * 60)
    print("ACCURACY")
    print("=" * 60)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Percentuale corretta: "
        f"{accuracy * 100:.2f}%"
    )

    print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def evaluate_classification_report(
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """
    Stampa precision, recall e F1-score
    per ogni classe.
    """

    target_names = [
        CLASS_NAMES[i]
        for i in range(len(CLASS_NAMES))
    ]

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=target_names,
            digits=4,
        )
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def evaluate_confusion_matrix(
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """
    Calcola e stampa la confusion matrix.
    """

    labels = list(CLASS_NAMES.keys())

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print()

    print(
        "          "
        + " ".join(
            f"{CLASS_NAMES[i]:>5}"
            for i in labels
        )
    )

    print()

    for i, row in enumerate(matrix):

        print(
            f"{CLASS_NAMES[i]:>5}     "
            + " ".join(
                f"{value:>5}"
                for value in row
            )
        )

    print()

    return matrix


# ============================================================
# ERRORI PER CLASSE
# ============================================================

def evaluate_errors_per_class(
    y_test: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """
    Calcola il numero di errori per ogni classe.
    """

    print("=" * 60)
    print("ERRORI PER CLASSE")
    print("=" * 60)

    for class_id, class_name in CLASS_NAMES.items():

        mask = y_test == class_id

        total = np.sum(mask)

        correct = np.sum(
            predictions[mask] == class_id
        )

        errors = total - correct

        accuracy = (
            correct / total
            if total > 0
            else 0
        )

        print(
            f"{class_name}: "
            f"{errors} errori / "
            f"{total} immagini "
            f"({accuracy * 100:.2f}% corrette)"
        )

    print()


# ============================================================
# CONFUSION MATRIX GRAFICA
# ============================================================

def plot_confusion_matrix(
    matrix: np.ndarray,
    dataset_name: str,
) -> None:
    """
    Mostra graficamente la confusion matrix.
    """

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            CLASS_NAMES[i]
            for i in range(len(CLASS_NAMES))
        ],
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        f"Confusion Matrix - {dataset_name} Test"
    )

    plt.xlabel(
        "Predicted label"
    )

    plt.ylabel(
        "True label"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# VALUTAZIONE DATASET
# ============================================================

def evaluate_dataset(
    model,
    dataset_name: str,
) -> None:
    """
    Esegue tutta la valutazione su un test set.
    """

    print()
    print("=" * 60)
    print(f"VALUTAZIONE {dataset_name}")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # CARICAMENTO
    # --------------------------------------------------------

    X_test, y_test = (
        load_test_dataset(
            dataset_name
        )
    )

    # --------------------------------------------------------
    # PREDIZIONI
    # --------------------------------------------------------

    predictions = make_predictions(
        model,
        X_test,
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    evaluate_accuracy(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    evaluate_classification_report(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    matrix = evaluate_confusion_matrix(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # ERRORI PER CLASSE
    # --------------------------------------------------------

    evaluate_errors_per_class(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # GRAFICO
    # --------------------------------------------------------

    plot_confusion_matrix(
        matrix,
        dataset_name,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VALUTAZIONE CLASSIFICATORE")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # CARICAMENTO MODELLO
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # VALUTAZIONE EMNIST
    # --------------------------------------------------------

    evaluate_dataset(
        model,
        "EMNIST",
    )

    # --------------------------------------------------------
    # VALUTAZIONE DIGITAL
    # --------------------------------------------------------

    evaluate_dataset(
        model,
        "DIGITAL",
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()