
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "letter_classifier.joblib"
IMAGE_DIR = BASE_DIR / "tests" / "immagini_lettere"

CLASS_NAMES = {
    0: "C",
    1: "D",
    2: "F",
    3: "S",
    4: "V",
    5: "X",
}


# ============================================================
# TROVA IMMAGINE
# ============================================================

def find_image(filename=None):

    if filename:
        image_path = Path(filename)

        if image_path.exists():
            return image_path.resolve()

        image_path = IMAGE_DIR / filename

        if image_path.exists():
            return image_path

        raise FileNotFoundError(
            f"Immagine non trovata: {filename}"
        )

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not images:
        raise FileNotFoundError(
            f"Nessuna immagine trovata in {IMAGE_DIR}"
        )

    return images[0]


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Prepara una lettera nera su sfondo bianco
    per il classificatore EMNIST.

    Restituisce:
        processed: immagine 28x28 normalizzata
        mask: maschera binaria
        bbox: bounding box della lettera
    """

    # --------------------------------------------------------
    # 1. Trova la lettera
    # --------------------------------------------------------

    # La lettera è nera, quindi i pixel con valore basso
    # appartengono alla lettera.
    _, mask = cv2.threshold(
        image,
        170,
        255,
        cv2.THRESH_BINARY_INV
    )

    # --------------------------------------------------------
    # 2. Trova la bounding box
    # --------------------------------------------------------

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        raise ValueError(
            "Non è stato possibile trovare la lettera."
        )

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    w = x_max - x_min + 1
    h = y_max - y_min + 1

    # --------------------------------------------------------
    # 3. Aggiungi un margine
    # --------------------------------------------------------

    margin = int(max(w, h) * 0.15)

    x1 = max(0, x_min - margin)
    y1 = max(0, y_min - margin)

    x2 = min(image.shape[1], x_max + margin + 1)
    y2 = min(image.shape[0], y_max + margin + 1)

    # --------------------------------------------------------
    # 4. Ritaglia la lettera
    # --------------------------------------------------------

    cropped = image[y1:y2, x1:x2]

    # --------------------------------------------------------
    # 5. Rendi il ritaglio quadrato
    # --------------------------------------------------------

    crop_h, crop_w = cropped.shape

    side = max(crop_h, crop_w)

    square = np.full(
        (side, side),
        255,
        dtype=np.uint8
    )

    offset_x = (side - crop_w) // 2
    offset_y = (side - crop_h) // 2

    square[
        offset_y:offset_y + crop_h,
        offset_x:offset_x + crop_w
    ] = cropped

    # --------------------------------------------------------
    # 6. Ridimensiona a 28x28
    # --------------------------------------------------------

    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # 7. Inverti i colori
    #
    # Immagine originale:
    #   nero  = lettera
    #   bianco = sfondo
    #
    # EMNIST:
    #   bianco = lettera
    #   nero = sfondo
    # --------------------------------------------------------

    processed = 255 - resized

    # --------------------------------------------------------
    # 8. Normalizza
    # --------------------------------------------------------

    processed = processed.astype(np.float32) / 255.0

    return processed, mask, (x1, y1, x2, y2)


# ============================================================
# VISUALIZZAZIONE
# ============================================================

def show_results(
    original,
    mask,
    processed,
    bbox,
    prediction,
    probabilities
):

    x1, y1, x2, y2 = bbox

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    # --------------------------------------------------------
    # Immagine originale + bounding box
    # --------------------------------------------------------

    axes[0].imshow(
        original,
        cmap="gray"
    )

    rect = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=2
    )

    axes[0].add_patch(rect)

    axes[0].set_title("Originale")
    axes[0].axis("off")

    # --------------------------------------------------------
    # Maschera
    # --------------------------------------------------------

    axes[1].imshow(
        mask,
        cmap="gray"
    )

    axes[1].set_title("Maschera")
    axes[1].axis("off")

    # --------------------------------------------------------
    # Input finale del modello
    # --------------------------------------------------------

    axes[2].imshow(
        processed,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        f"Input modello → {prediction}"
    )

    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Probabilità
    # --------------------------------------------------------

    print("\nProbabilità:")

    for class_id, probability in enumerate(probabilities):

        print(
            f"{CLASS_NAMES[class_id]}: "
            f"{probability * 100:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Caricamento modello...")

    model = joblib.load(MODEL_PATH)

    filename = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    image_path = find_image(filename)

    print(
        f"Immagine utilizzata: {image_path}"
    )

    # --------------------------------------------------------
    # Carica immagine
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Impossibile leggere l'immagine: {image_path}"
        )

    print(
        f"Dimensioni originali: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    processed, mask, bbox = preprocess_image(
        image
    )

    # --------------------------------------------------------
    # Predizione
    # --------------------------------------------------------

    X = processed.reshape(1, 784)

    probabilities = model.predict_proba(X)[0]

    predicted_class = model.predict(X)[0]

    prediction = CLASS_NAMES[predicted_class]

    confidence = probabilities[predicted_class]

    print("\n--------------------------------")
    print(f"Predizione: {prediction}")
    print(
        f"Confidenza: {confidence * 100:.2f}%"
    )
    print("--------------------------------")

    # --------------------------------------------------------
    # Visualizzazione
    # --------------------------------------------------------

    show_results(
        image,
        mask,
        processed,
        bbox,
        prediction,
        probabilities
    )



def predict_image(image_path, model):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Impossibile leggere l'immagine: {image_path}"
        )

    processed, _, _ = preprocess_image(image)

    X = processed.reshape(1, 784)

    probabilities = model.predict_proba(X)[0]

    predicted_class = model.predict(X)[0]

    letter = CLASS_NAMES[predicted_class]

    confidence = probabilities[predicted_class]

    return letter, confidence


if __name__ == "__main__":
    main()