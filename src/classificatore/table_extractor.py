import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURAZIONE
# ============================================================

CELL_MARGIN = 0.10


# ============================================================
# ORDINE DEI PUNTI
# ============================================================

def order_points(points):
    """
    Ordina i quattro vertici come:

        top-left     top-right
        bottom-left  bottom-right
    """

    points = np.array(points, dtype=np.float32)

    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).flatten()

    ordered[0] = points[np.argmin(sums)]   # top-left
    ordered[1] = points[np.argmin(diffs)]  # top-right
    ordered[2] = points[np.argmax(sums)]   # bottom-right
    ordered[3] = points[np.argmax(diffs)]  # bottom-left

    return ordered


# ============================================================
# PREPROCESSING
# ============================================================

def create_binary(image):
    """
    Crea una versione binaria dell'immagine
    utile per rilevare la griglia.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        5
    )

    return binary


# ============================================================
# RILEVAMENTO TABELLA
# ============================================================

def find_table_corners(image):
    """
    Cerca il quadrilatero esterno della tabella.

    Restituisce i quattro vertici oppure None.
    """

    binary = create_binary(image)

    height, width = binary.shape

    horizontal_size = max(
        20,
        width // 15
    )

    vertical_size = max(
        20,
        height // 15
    )

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

    grid = cv2.bitwise_or(
        horizontal,
        vertical
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
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

    image_area = width * height

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.05:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approximation) == 4:

            candidates.append(
                (
                    area,
                    approximation.reshape(4, 2)
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return order_points(
        candidates[0][1]
    )


# ============================================================
# CORREZIONE PROSPETTIVA
# ============================================================

def rectify_table(image, corners):
    """
    Corregge la prospettiva della fotografia.

    La tabella viene trasformata in un'immagine quadrata.
    """

    corners = order_points(corners)

    top_left = corners[0]
    top_right = corners[1]
    bottom_right = corners[2]
    bottom_left = corners[3]

    width_top = np.linalg.norm(
        top_right - top_left
    )

    width_bottom = np.linalg.norm(
        bottom_right - bottom_left
    )

    height_left = np.linalg.norm(
        bottom_left - top_left
    )

    height_right = np.linalg.norm(
        bottom_right - top_right
    )

    width = int(
        max(
            width_top,
            width_bottom
        )
    )

    height = int(
        max(
            height_left,
            height_right
        )
    )

    # La griglia è quadrata.
    size = max(
        width,
        height
    )

    destination = np.array(
        [
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        corners,
        destination
    )

    rectified = cv2.warpPerspective(
        image,
        matrix,
        (size, size)
    )

    return rectified


# ============================================================
# RILEVAMENTO LINEE
# ============================================================

def cluster_positions(positions, max_distance=10):
    """
    Raggruppa coordinate vicine appartenenti alla stessa linea.

    Esempio:

        [100, 101, 102, 103, 150, 151]

    diventa:

        [101.5, 150.5]
    """

    if len(positions) == 0:
        return []

    positions = sorted(
        positions
    )

    groups = []
    current_group = [
        positions[0]
    ]

    for position in positions[1:]:

        if (
            position
            - current_group[-1]
            <= max_distance
        ):
            current_group.append(
                position
            )

        else:
            groups.append(
                current_group
            )

            current_group = [
                position
            ]

    groups.append(
        current_group
    )

    return [
        int(round(np.mean(group)))
        for group in groups
    ]


def detect_grid_lines(table):
    """
    Rileva le linee orizzontali e verticali della griglia.

    Le posizioni servono esclusivamente per determinare
    quante celle sono presenti.
    """

    binary = create_binary(
        table
    )

    height, width = binary.shape

    # --------------------------------------------------------
    # Linee orizzontali
    # --------------------------------------------------------

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(15, width // 20),
            1
        )
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    horizontal_projection = np.sum(
        horizontal > 0,
        axis=1
    )

    horizontal_positions = np.where(
        horizontal_projection
        > width * 0.25
    )[0]

    horizontal_lines = cluster_positions(
        horizontal_positions
    )

    # --------------------------------------------------------
    # Linee verticali
    # --------------------------------------------------------

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            1,
            max(15, height // 20)
        )
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    vertical_projection = np.sum(
        vertical > 0,
        axis=0
    )

    vertical_positions = np.where(
        vertical_projection
        > height * 0.25
    )[0]

    vertical_lines = cluster_positions(
        vertical_positions
    )

    return (
        horizontal_lines,
        vertical_lines
    )


# ============================================================
# DETERMINAZIONE DIMENSIONE GRIGLIA
# ============================================================

def detect_grid_size(table):
    """
    Determina automaticamente N per una griglia N x N.

    Una griglia N x N ha N+1 linee per direzione.
    """

    horizontal_lines, vertical_lines = (
        detect_grid_lines(table)
    )

    print(
        f"Linee orizzontali trovate: "
        f"{len(horizontal_lines)}"
    )

    print(
        f"Linee verticali trovate: "
        f"{len(vertical_lines)}"
    )

    horizontal_cells = (
        len(horizontal_lines) - 1
    )

    vertical_cells = (
        len(vertical_lines) - 1
    )

    print(
        f"Celle orizzontali: "
        f"{horizontal_cells}"
    )

    print(
        f"Celle verticali: "
        f"{vertical_cells}"
    )

    if horizontal_cells <= 0:
        return None

    if vertical_cells <= 0:
        return None

    if horizontal_cells != vertical_cells:

        raise ValueError(
            "La griglia rilevata non è quadrata: "
            f"{horizontal_cells}x{vertical_cells}."
        )

    return horizontal_cells


# ============================================================
# ESTRAZIONE CELLE
# ============================================================

def extract_cells(table, grid_size, output_dir):
    """
    Divide la tabella rettificata in N x N celle.

    Le celle vengono estratte geometricamente,
    lasciando un margine interno per evitare
    che le linee della griglia vengano classificate.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    height, width = table.shape[:2]

    cell_width = (
        width / grid_size
    )

    cell_height = (
        height / grid_size
    )

    print(
        f"Griglia rilevata: "
        f"{grid_size} x {grid_size}"
    )

    count = 0

    for row in range(grid_size):

        for col in range(grid_size):

            x1 = int(
                col * cell_width
            )

            y1 = int(
                row * cell_height
            )

            x2 = int(
                (col + 1)
                * cell_width
            )

            y2 = int(
                (row + 1)
                * cell_height
            )

            # ------------------------------------------------
            # Margine interno
            # ------------------------------------------------

            margin_x = int(
                (x2 - x1)
                * CELL_MARGIN
            )

            margin_y = int(
                (y2 - y1)
                * CELL_MARGIN
            )

            x1_crop = x1 + margin_x
            y1_crop = y1 + margin_y

            x2_crop = x2 - margin_x
            y2_crop = y2 - margin_y

            cell = table[
                y1_crop:y2_crop,
                x1_crop:x2_crop
            ]

            output_path = (
                output_dir
                / f"cell_{row:02d}_{col:02d}.png"
            )

            cv2.imwrite(
                str(output_path),
                cell
            )

            count += 1

    print(
        f"Estratte {count} celle."
    )


# ============================================================
# FUNZIONE PUBBLICA
# ============================================================

def extract_table_cells(
    image_path,
    output_dir="cells"
):
    """
    Pipeline completa:

        immagine
          ↓
        rilevamento tabella
          ↓
        correzione prospettiva
          ↓
        rilevamento dimensione N
          ↓
        estrazione N x N celle

    Questa è la funzione utilizzata da predict_table.py.
    """

    image_path = Path(
        image_path
    )

    output_dir = Path(
        output_dir
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Impossibile leggere "
            f"l'immagine: {image_path}"
        )

    print(
        "Immagine caricata."
    )

    print(
        "Cerco la tabella..."
    )

    corners = find_table_corners(
        image
    )

    if corners is None:

        raise ValueError(
            "Tabella non rilevata."
        )

    print(
        "Tabella trovata."
    )

    print(
        "Correggo la prospettiva..."
    )

    table = rectify_table(
        image,
        corners
    )

    rectified_path = (
        output_dir
        / "table_rectified.png"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(rectified_path),
        table
    )

    print(
        "Tabella rettificata salvata."
    )

    print(
        "Cerco le linee della griglia..."
    )

    grid_size = detect_grid_size(
        table
    )

    if grid_size is None:

        raise ValueError(
            "Impossibile determinare "
            "la dimensione della griglia."
        )

    extract_cells(
        table,
        grid_size,
        output_dir
    )

    print(
        "Operazione completata."
    )

    print(
        f"Output: {output_dir}"
    )

    return grid_size


# ============================================================
# ESECUZIONE DIRETTA
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Uso: python table_extractor.py "
            "<immagine>"
        )

        sys.exit(1)

    extract_table_cells(
        sys.argv[1]
    )