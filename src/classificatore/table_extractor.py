
import cv2
import numpy as np
import os
import sys


# ============================================================
# CONFIGURAZIONE
# ============================================================

OUTPUT_DIR = "cells_test_1"

# Percentuale della cella da eliminare lungo i bordi.
CELL_MARGIN = 0.08


# ============================================================
# UTILITY
# ============================================================

def clear_output_dir(output_dir):
    """
    Svuota la cartella di output prima di una nuova estrazione.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(output_dir):

        path = os.path.join(
            output_dir,
            filename
        )

        if os.path.isfile(path):
            os.remove(path)


def cluster_positions(values, tolerance):
    """
    Raggruppa coordinate molto vicine.
    """

    if len(values) == 0:
        return []

    values = sorted(values)

    clusters = [[values[0]]]

    for value in values[1:]:

        if abs(value - np.mean(clusters[-1])) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    return [
        int(round(np.mean(c)))
        for c in clusters
    ]


# ============================================================
# PREPROCESSING
# ============================================================

def create_binary(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    return binary


# ============================================================
# RILEVAMENTO DELLE LINEE
# ============================================================

def detect_grid_lines(image):

    binary = create_binary(image)

    h, w = binary.shape

    horizontal_length = max(
        15,
        w // 20
    )

    vertical_length = max(
        15,
        h // 20
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_length, 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_length)
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    horizontal_projection = np.sum(
        horizontal > 0,
        axis=1
    )

    vertical_projection = np.sum(
        vertical > 0,
        axis=0
    )

    horizontal_threshold = w * 0.30
    vertical_threshold = h * 0.30

    horizontal_indices = np.where(
        horizontal_projection > horizontal_threshold
    )[0]

    vertical_indices = np.where(
        vertical_projection > vertical_threshold
    )[0]

    horizontal_positions = cluster_positions(
        horizontal_indices.tolist(),
        tolerance=max(5, h // 150)
    )

    vertical_positions = cluster_positions(
        vertical_indices.tolist(),
        tolerance=max(5, w // 150)
    )

    return (
        horizontal_positions,
        vertical_positions
    )


# ============================================================
# FALLBACK: BORDI DELL'IMMAGINE
# ============================================================

def add_missing_borders(
    horizontal,
    vertical,
    image
):

    h, w = image.shape[:2]

    horizontal = list(horizontal)
    vertical = list(vertical)

    if len(horizontal) < 2:

        if not horizontal or horizontal[0] > 0:
            horizontal.insert(0, 0)

        horizontal.append(h - 1)

    if len(vertical) < 2:

        if not vertical or vertical[0] > 0:
            vertical.insert(0, 0)

        vertical.append(w - 1)

    return horizontal, vertical


# ============================================================
# VALIDAZIONE DELLA GRIGLIA
# ============================================================

def validate_grid(
    horizontal,
    vertical
):

    rows = len(horizontal) - 1
    cols = len(vertical) - 1

    if rows < 1 or cols < 1:
        return False

    if rows != cols:
        return False

    if rows > 100:
        return False

    return True


# ============================================================
# VISUALIZZAZIONE DEL RILEVAMENTO
# ============================================================

def create_debug_image(
    image,
    horizontal,
    vertical
):
    """
    Crea un'immagine sulla quale vengono disegnate
    le linee rilevate.

    Verde = linee orizzontali
    Rosso = linee verticali
    """

    debug = image.copy()

    h, w = debug.shape[:2]

    for y in horizontal:

        cv2.line(
            debug,
            (0, y),
            (w - 1, y),
            (0, 255, 0),
            2
        )

    for x in vertical:

        cv2.line(
            debug,
            (x, 0),
            (x, h - 1),
            (0, 0, 255),
            2
        )

    return debug


# ============================================================
# VISUALIZZAZIONE CELLE
# ============================================================

def create_cells_preview(
    image,
    horizontal,
    vertical
):
    """
    Crea un'immagine di anteprima con i rettangoli
    delle celle.
    """

    preview = image.copy()

    rows = len(horizontal) - 1
    cols = len(vertical) - 1

    for row in range(rows):

        for col in range(cols):

            x1 = vertical[col]
            x2 = vertical[col + 1]

            y1 = horizontal[row]
            y2 = horizontal[row + 1]

            cv2.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                preview,
                f"{row},{col}",
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

    return preview


# ============================================================
# ESTRAZIONE DELLE CELLE
# ============================================================

def extract_cells(
    image,
    horizontal,
    vertical,
    output_dir=OUTPUT_DIR
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    h, w = image.shape[:2]

    rows = len(horizontal) - 1
    cols = len(vertical) - 1

    print(
        f"Griglia rilevata: {rows} x {cols}"
    )

    cell_paths = []

    for row in range(rows):

        y1 = horizontal[row]
        y2 = horizontal[row + 1]

        for col in range(cols):

            x1 = vertical[col]
            x2 = vertical[col + 1]

            cell_width = x2 - x1
            cell_height = y2 - y1

            margin_x = int(
                cell_width * CELL_MARGIN
            )

            margin_y = int(
                cell_height * CELL_MARGIN
            )

            crop_x1 = max(
                0,
                x1 + margin_x
            )

            crop_x2 = min(
                w,
                x2 - margin_x
            )

            crop_y1 = max(
                0,
                y1 + margin_y
            )

            crop_y2 = min(
                h,
                y2 - margin_y
            )

            cell = image[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            filename = (
                f"cell_{row:02d}_{col:02d}.png"
            )

            path = os.path.join(
                output_dir,
                filename
            )

            success = cv2.imwrite(
                path,
                cell
            )

            if not success:
                raise RuntimeError(
                    f"Impossibile salvare: {path}"
                )

            cell_paths.append(path)

    print(
        f"Estratte {rows * cols} celle."
    )

    return cell_paths


# ============================================================
# PIPELINE COMPLETA
# ============================================================

def extract_table_cells(
    image_path,
    output_dir=OUTPUT_DIR
):

    image_path = str(image_path)

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            f"Impossibile aprire '{image_path}'"
        )

    clear_output_dir(
        output_dir
    )

    print(
        "Immagine caricata."
    )

    # --------------------------------------------------------
    # 1. TROVA LE LINEE
    # --------------------------------------------------------

    print(
        "Cerco le linee della griglia..."
    )

    horizontal, vertical = (
        detect_grid_lines(image)
    )

    print(
        f"Linee orizzontali trovate: "
        f"{len(horizontal)}"
    )

    print(
        f"Linee verticali trovate: "
        f"{len(vertical)}"
    )

    # --------------------------------------------------------
    # 2. AGGIUNGE EVENTUALI BORDI
    # --------------------------------------------------------

    horizontal, vertical = (
        add_missing_borders(
            horizontal,
            vertical,
            image
        )
    )

    # --------------------------------------------------------
    # 3. ORDINA
    # --------------------------------------------------------

    horizontal = sorted(horizontal)
    vertical = sorted(vertical)

    # --------------------------------------------------------
    # 4. SALVA DEBUG
    # --------------------------------------------------------

    debug = create_debug_image(
        image,
        horizontal,
        vertical
    )

    debug_path = os.path.join(
        output_dir,
        "debug_lines.png"
    )

    cv2.imwrite(
        debug_path,
        debug
    )

    preview = create_cells_preview(
        image,
        horizontal,
        vertical
    )

    preview_path = os.path.join(
        output_dir,
        "debug_cells.png"
    )

    cv2.imwrite(
        preview_path,
        preview
    )

    # --------------------------------------------------------
    # 5. VALIDAZIONE
    # --------------------------------------------------------

    if not validate_grid(
        horizontal,
        vertical
    ):

        raise ValueError(
            "La griglia rilevata non è valida. "
            f"Intervalli orizzontali: "
            f"{len(horizontal) - 1}, "
            f"intervalli verticali: "
            f"{len(vertical) - 1}."
        )

    # --------------------------------------------------------
    # 6. ESTRAZIONE
    # --------------------------------------------------------

    return extract_cells(
        image,
        horizontal,
        vertical,
        output_dir
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Uso:\n"
            "    python3 table_extractor_1.py immagine.jpg"
        )

        sys.exit(1)

    try:

        extract_table_cells(
            sys.argv[1]
        )

    except Exception as e:

        print(
            f"\nErrore: {e}"
        )

        sys.exit(1)

    print()
    print(
        "Operazione completata."
    )

    print(
        f"Output: {OUTPUT_DIR}/"
    )

    print()
    print(
        "File di debug:"
    )

    print(
        f"  {OUTPUT_DIR}/debug_lines.png"
    )

    print(
        f"  {OUTPUT_DIR}/debug_cells.png"
    )


if __name__ == "__main__":
    main()
