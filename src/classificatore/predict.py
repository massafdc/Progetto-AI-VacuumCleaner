from pathlib import Path

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
# TROVA LA PRIMA IMMAGINE
# ============================================================

def find_first_image():
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
    Restituisce:

    - processed: immagine 28x28 grayscale pronta per il modello
    - mask: maschera binaria usata per trovare la lettera
    - bbox: bounding box trovata
    """

    # --------------------------------------------------------
    # 1. Correzione dell'illuminazione
    # --------------------------------------------------------

    # Sfocatura molto grande = stima dello sfondo/illuminazione
    background = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=25
    )

    # Evita divisioni per zero
    background = np.maximum(background, 1)

    # Normalizzazione dell'illuminazione
    normalized = cv2.divide(
        image,
        background,
        scale=255
    )

    # --------------------------------------------------------
    # 2. Maschera binaria per trovare la X
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        normalized,
        (5, 5),
        0
    )

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # 3. Elimina piccoli rumori
    # --------------------------------------------------------

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Chiude eventuali piccoli buchi nella X
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # --------------------------------------------------------
    # 4. Trova i pixel appartenenti alla lettera
    # --------------------------------------------------------

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        raise ValueError(
            "Non è stato possibile trovare la lettera nell'immagine."
        )

    # --------------------------------------------------------
    # 5. Bounding box
    # --------------------------------------------------------

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    w = x_max - x_min + 1
    h = y_max - y_min + 1

    # Margine attorno alla lettera
    margin = int(max(w, h) * 0.30)

    x1 = max(0, x_min - margin)
    y1 = max(0, y_min - margin)

    x2 = min(image.shape[1], x_max + margin + 1)
    y2 = min(image.shape[0], y_max + margin + 1)

    # --------------------------------------------------------
    # 6. Crop dell'immagine originale NORMALIZZATA
    # --------------------------------------------------------

    cropped = normalized[y1:y2, x1:x2]

    # --------------------------------------------------------
    # 7. Rendi il crop quadrato
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
    # 8. Resize a 28x28
    # --------------------------------------------------------

    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # 9. Inversione:
    #
    # foto originale:
    # nero = lettera
    # bianco = sfondo
    #
    # dataset:
    # bianco = lettera
    # nero = sfondo
    # --------------------------------------------------------

    processed = 255 - resized

    # Normalizzazione 0-1
    processed = processed.astype(np.float32) / 255.0

    return processed, mask, (x1, y1, x2, y2), normalized


# ============================================================
# VISUALIZZAZIONE
# ============================================================

def show_results(
    original,
    normalized,
    mask,
    processed,
    bbox,
    prediction,
    probabilities
):

    x1, y1, x2, y2 = bbox

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # --------------------------------------------------------
    # Immagine originale + bounding box
    # --------------------------------------------------------

    axes[0].imshow(original, cmap="gray")

    rect = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=2
    )

    axes[0].add_patch(rect)

    axes[0].set_title("Originale + bounding box")
    axes[0].axis("off")

    # --------------------------------------------------------
    # Immagine dopo correzione illuminazione
    # --------------------------------------------------------

    axes[1].imshow(normalized, cmap="gray")
    axes[1].set_title("Illuminazione corretta")
    axes[1].axis("off")

    # --------------------------------------------------------
    # Maschera usata per trovare la X
    # --------------------------------------------------------

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Maschera binaria")
    axes[2].axis("off")

    # --------------------------------------------------------
    # Immagine finale 28x28
    # --------------------------------------------------------

    axes[3].imshow(
        processed.reshape(28, 28),
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[3].set_title(
        f"Input modello → {prediction}"
    )

    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Statistiche
    # --------------------------------------------------------

    print("\nStatistiche immagine finale:")
    print(f"min:  {processed.min():.4f}")
    print(f"max:  {processed.max():.4f}")
    print(f"mean: {processed.mean():.4f}")

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

    image_path = find_first_image()

    print(f"Immagine utilizzata: {image_path}")

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

    processed, mask, bbox, normalized = preprocess_image(
        image
    )

    # --------------------------------------------------------
    # Predizione
    # --------------------------------------------------------

    X = processed.reshape(1, 784)

    probabilities = model.predict_proba(X)[0]

    predicted_class = model.predict(X)[0]

    prediction = CLASS_NAMES[predicted_class]

    print("\n--------------------------------")
    print(f"Predizione: {prediction}")
    print(
        f"Confidenza: "
        f"{probabilities[predicted_class] * 100:.2f}%"
    )
    print("--------------------------------")

    # --------------------------------------------------------
    # Visualizzazione
    # --------------------------------------------------------

    show_results(
        image,
        normalized,
        mask,
        processed,
        bbox,
        prediction,
        probabilities
    )


if __name__ == "__main__":
    main()
