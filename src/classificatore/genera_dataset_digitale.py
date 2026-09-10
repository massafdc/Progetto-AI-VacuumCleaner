from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = BASE_DIR / "data" / "digital" / "processed"

IMAGE_SIZE = 28

SAMPLES_PER_CLASS = 5000

RANDOM_SEED = 42

CLASS_NAMES = {
    0: "C",
    1: "D",
    2: "F",
    3: "S",
    4: "V",
    5: "X",
}

# Font comuni disponibili su macOS.
# Lo script utilizzerà solo quelli effettivamente presenti.
FONT_PATHS = [
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Times.ttc",
    "/System/Library/Fonts/Courier.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/georgia.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "C:/Windows/Fonts/trebuc.ttf",
]


# ============================================================
# FONT
# ============================================================

def get_available_fonts() -> list[Path]:
    """
    Restituisce i font disponibili tra quelli configurati.
    """

    fonts = [
        Path(path)
        for path in FONT_PATHS
        if Path(path).exists()
    ]

    if not fonts:
        raise RuntimeError(
            "Nessun font configurato è stato trovato."
        )

    return fonts


# ============================================================
# GENERAZIONE IMMAGINE
# ============================================================

def generate_letter_image(
    letter: str,
    font_path: Path,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Genera una singola immagine 28x28 della lettera indicata.

    L'immagine viene creata in scala di grigi:
        0   = nero
        255 = bianco

    Vengono applicate piccole variazioni di:
        - dimensione
        - posizione
        - rotazione
        - scala
    """

    # Canvas più grande del risultato finale.
    canvas_size = 64

    image = Image.new(
        "L",
        (canvas_size, canvas_size),
        0,
    )

    draw = ImageDraw.Draw(image)

    # Dimensione casuale del carattere.
    font_size = int(
        rng.integers(20, 31)
    )

    font = ImageFont.truetype(
        str(font_path),
        font_size,
    )

    # Bounding box della lettera.
    bbox = draw.textbbox(
        (0, 0),
        letter,
        font=font,
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Piccole variazioni di posizione.
    x = (canvas_size - text_width) // 2
    y = (canvas_size - text_height) // 2

    x += int(rng.integers(-4, 5))
    y += int(rng.integers(-4, 5))

    draw.text(
        (x, y),
        letter,
        fill=255,
        font=font,
    )

    # Piccola rotazione.
    angle = float(
        rng.uniform(-5.0, 5.0)
    )

    image = image.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=0,
    )

    # Ridimensionamento a 28x28.
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    # Conversione NumPy.
    array = np.asarray(
        image,
        dtype=np.float32,
    )

    # Normalizzazione [0,255] -> [0,1].
    array /= 255.0

    return array


# ============================================================
# DATASET
# ============================================================

def generate_dataset() -> None:
    """
    Genera il dataset digitale completo.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fonts = get_available_fonts()

    print("Font disponibili:")

    for font in fonts:
        print(f"  - {font}")

    print()

    total_samples = (
        len(CLASS_NAMES) * SAMPLES_PER_CLASS
    )

    X = np.empty(
        (
            total_samples,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        dtype=np.float32,
    )

    y = np.empty(
        total_samples,
        dtype=np.int64,
    )

    index = 0

    for class_id, letter in CLASS_NAMES.items():

        print(
            f"Generazione classe {class_id} "
            f"({letter})..."
        )

        for _ in range(SAMPLES_PER_CLASS):

            font_path = fonts[
                rng.integers(0, len(fonts))
            ]

            image = generate_letter_image(
                letter,
                font_path,
                rng,
            )

            X[index] = image
            y[index] = class_id

            index += 1

    # Mescoliamo il dataset.
    permutation = rng.permutation(
        total_samples
    )

    X = X[permutation]
    y = y[permutation]

    # Salvataggio.
    np.save(
        OUTPUT_DIR / "X.npy",
        X,
    )

    np.save(
        OUTPUT_DIR / "y.npy",
        y,
    )

    print()
    print("Dataset digitale generato.")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print()
    print(f"Salvato in: {OUTPUT_DIR}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_dataset()