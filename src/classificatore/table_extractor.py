import cv2
import numpy as np
import os
import sys


# ============================================================
# CONFIGURAZIONE
# ============================================================

OUTPUT_DIR = "cells"

# Percentuale della cella da eliminare lungo i bordi.
# Serve per evitare che le linee della tabella entrino
# nell'immagine che verrà passata al classificatore.
CELL_MARGIN = 0.08


# ============================================================
# UTILITY
# ============================================================

def cluster_positions(values, tolerance):
    """
    Raggruppa coordinate molto vicine.

    Esempio:
        [100, 102, 104, 250, 252]

    diventa:
        [102, 251]
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

    return [int(round(np.mean(c))) for c in clusters]


def order_points(points):
    """
    Ordina 4 punti:
        top-left
        top-right
        bottom-right
        bottom-left
    """

    points = np.array(points, dtype=np.float32)

    s = points.sum(axis=1)
    d = np.diff(points, axis=1).flatten()

    top_left = points[np.argmin(s)]
    bottom_right = points[np.argmax(s)]
    top_right = points[np.argmin(d)]
    bottom_left = points[np.argmax(d)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32
    )


# ============================================================
# PREPROCESSING
# ============================================================

def create_binary(image):
    """
    Crea un'immagine binaria robusta sia per immagini digitali
    sia per fotografie.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Riduzione del rumore
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold:
    # funziona meglio del semplice threshold quando la foto
    # ha zone più scure o più illuminate.
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
# RILEVAMENTO DELLA TABELLA
# ============================================================

def find_table_corners(image):
    """
    Cerca il quadrilatero esterno della tabella.

    Se non viene trovato, restituisce None.
    """

    binary = create_binary(image)

    h, w = binary.shape

    # Kernel abbastanza grande da collegare le linee della griglia.
    horizontal_size = max(10, w // 30)
    vertical_size = max(10, h // 30)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_size, 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_size)
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

    # Unisce le linee.
    grid = cv2.bitwise_or(horizontal, vertical)

    # Piccola chiusura per colmare interruzioni.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    grid = cv2.morphologyEx(
        grid,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        grid,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = w * h

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # La tabella deve occupare una parte significativa
        # dell'immagine.
        if area < image_area * 0.05:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True
        )

        if len(approx) == 4:

            candidates.append(
                (area, approx.reshape(4, 2))
            )

    if not candidates:
        return None

    # Prende il quadrilatero più grande.
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return candidates[0][1]


# ============================================================
# CORREZIONE PROSPETTIVA
# ============================================================

def rectify_table(image, corners):
    """
    Trasforma la tabella in un quadrato.
    """

    corners = order_points(corners)

    tl, tr, br, bl = corners

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    size = max(width, height)

    destination = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        corners,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (size, size)
    )

    return warped


# ============================================================
# RILEVAMENTO DELLE LINEE
# ============================================================

def detect_grid_lines(image):
    """
    Trova automaticamente tutte le linee verticali e orizzontali
    della griglia.
    """

    binary = create_binary(image)

    h, w = binary.shape

    # --------------------------------------------------------
    # LINEE ORIZZONTALI
    # --------------------------------------------------------

    horizontal_length = max(15, w // 20)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_length, 1)
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # --------------------------------------------------------
    # LINEE VERTICALI
    # --------------------------------------------------------

    vertical_length = max(15, h // 20)

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_length)
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    # --------------------------------------------------------
    # PROIEZIONI
    # --------------------------------------------------------

    horizontal_projection = np.sum(
        horizontal > 0,
        axis=1
    )

    vertical_projection = np.sum(
        vertical > 0,
        axis=0
    )

    # Una linea deve occupare una percentuale significativa
    # della larghezza/altezza.
    horizontal_threshold = w * 0.25
    vertical_threshold = h * 0.25

    horizontal_indices = np.where(
        horizontal_projection > horizontal_threshold
    )[0]

    vertical_indices = np.where(
        vertical_projection > vertical_threshold
    )[0]

    # --------------------------------------------------------
    # RAGGRUPPA LE LINEE SPESSE
    # --------------------------------------------------------

    horizontal_positions = cluster_positions(
        horizontal_indices.tolist(),
        tolerance=max(5, h // 150)
    )

    vertical_positions = cluster_positions(
        vertical_indices.tolist(),
        tolerance=max(5, w // 150)
    )

    return horizontal_positions, vertical_positions


# ============================================================
# FALLBACK: BORDI DELL'IMMAGINE
# ============================================================

def add_missing_borders(horizontal, vertical, image):
    """
    Se le linee esterne non sono state rilevate perfettamente,
    considera anche i bordi della tabella rettificata.
    """

    h, w = image.shape[:2]

    tolerance = max(10, min(h, w) // 50)

    horizontal = list(horizontal)
    vertical = list(vertical)

    # Se manca una linea molto vicina al bordo,
    # aggiungiamo il bordo.
    if not horizontal or horizontal[0] > tolerance:
        horizontal.insert(0, 0)

    if not horizontal or abs(horizontal[-1] - h) > tolerance:
        horizontal.append(h - 1)

    if not vertical or vertical[0] > tolerance:
        vertical.insert(0, 0)

    if not vertical or abs(vertical[-1] - w) > tolerance:
        vertical.append(w - 1)

    return horizontal, vertical


# ============================================================
# VALIDAZIONE DELLA GRIGLIA
# ============================================================

def validate_grid(horizontal, vertical):
    """
    Controlla che la griglia abbia senso.

    Una tabella quadrata deve avere lo stesso numero di
    intervalli orizzontali e verticali.
    """

    rows = len(horizontal) - 1
    cols = len(vertical) - 1

    if rows < 1 or cols < 1:
        return False

    if rows != cols:
        return False

    # Evita risultati assurdi causati da rumore.
    if rows > 100:
        return False

    return True


# ============================================================
# ESTRAZIONE DELLE CELLE
# ============================================================

def extract_cells(image, horizontal, vertical, output_dir=OUTPUT_DIR):
    """
    Estrae tutte le celle nell'ordine:

        riga 0:
            colonna 0
            colonna 1
            ...

        riga 1:
            colonna 0
            ...

    Restituisce la lista dei percorsi delle celle create.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    h, w = image.shape[:2]

    rows = len(horizontal) - 1
    cols = len(vertical) - 1

    print(f"Griglia rilevata: {rows} x {cols}")

    cell_paths = []

    for row in range(rows):

        y1 = horizontal[row]
        y2 = horizontal[row + 1]

        for col in range(cols):

            x1 = vertical[col]
            x2 = vertical[col + 1]

            cell_width = x2 - x1
            cell_height = y2 - y1

            # ------------------------------------------------
            # MARGINE INTERNO
            # ------------------------------------------------

            margin_x = int(
                cell_width * CELL_MARGIN
            )

            margin_y = int(
                cell_height * CELL_MARGIN
            )

            crop_x1 = x1 + margin_x
            crop_x2 = x2 - margin_x

            crop_y1 = y1 + margin_y
            crop_y2 = y2 - margin_y

            # Sicurezza
            crop_x1 = max(0, crop_x1)
            crop_y1 = max(0, crop_y1)
            crop_x2 = min(w, crop_x2)
            crop_y2 = min(h, crop_y2)

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
                    f"Impossibile salvare la cella: {path}"
                )

            cell_paths.append(path)

    print(
        f"Estratte {rows * cols} celle."
    )

    return cell_paths


# ============================================================
# PIPELINE COMPLETA
# ============================================================

def extract_table_cells(image_path, output_dir=OUTPUT_DIR):
    """
    Esegue l'intera pipeline di estrazione della tabella:

        immagine
            ↓
        ricerca tabella
            ↓
        correzione prospettica
            ↓
        rilevamento linee
            ↓
        aggiunta bordi mancanti
            ↓
        validazione griglia
            ↓
        estrazione celle

    Restituisce:
        lista dei percorsi delle celle estratte.
    """

    image_path = str(image_path)

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Impossibile aprire '{image_path}'"
        )

    print("Immagine caricata.")

    # --------------------------------------------------------
    # 1. CERCA LA TABELLA
    # --------------------------------------------------------

    print("Cerco la tabella...")

    corners = find_table_corners(image)

    if corners is not None:

        print("Tabella trovata.")

        # ----------------------------------------------------
        # 2. CORREGGE LA PROSPETTIVA
        # ----------------------------------------------------

        print("Correggo la prospettiva...")

        table = rectify_table(
            image,
            corners
        )

    else:

        print(
            "Bordo esterno non trovato."
        )

        print(
            "Uso direttamente l'immagine."
        )

        table = image

    # --------------------------------------------------------
    # SALVA TABELLA RADDRIZZATA
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    rectified_path = os.path.join(
        output_dir,
        "table_rectified.png"
    )

    success = cv2.imwrite(
        rectified_path,
        table
    )

    if not success:
        raise RuntimeError(
            f"Impossibile salvare '{rectified_path}'"
        )

    # --------------------------------------------------------
    # 3. TROVA LE LINEE
    # --------------------------------------------------------

    print("Cerco le linee della griglia...")

    horizontal, vertical = detect_grid_lines(
        table
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
    # 4. AGGIUNGE EVENTUALI BORDI MANCANTI
    # --------------------------------------------------------

    horizontal, vertical = add_missing_borders(
        horizontal,
        vertical,
        table
    )

    # --------------------------------------------------------
    # 5. ORDINA
    # --------------------------------------------------------

    horizontal = sorted(horizontal)
    vertical = sorted(vertical)

    # --------------------------------------------------------
    # 6. CONTROLLA LA GRIGLIA
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
    # 7. ESTRAE LE CELLE
    # --------------------------------------------------------

    return extract_cells(
        table,
        horizontal,
        vertical,
        output_dir=output_dir
    )


# ============================================================
# PROGRAMMA PRINCIPALE
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Uso:\n"
            "    python3 table_extractor.py immagine.jpg"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    try:

        extract_table_cells(
            image_path,
            output_dir=OUTPUT_DIR
        )

    except Exception as e:

        print(
            f"\nErrore: {e}"
        )

        sys.exit(1)

    print()
    print("Operazione completata.")
    print(
        f"Output: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()