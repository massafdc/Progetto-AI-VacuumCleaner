from pathlib import Path
import shutil
import subprocess
import sys

import joblib

from .predict import predict_image, MODEL_PATH


BASE_DIR = Path(__file__).resolve().parents[2]
CELLS_DIR = BASE_DIR / "cells"


def main():

    if len(sys.argv) != 2:
        print("Uso:")
        print("python3 -m src.classificatore.predict_table percorso/tabella.png")
        sys.exit(1)

    table_path = Path(sys.argv[1])

    if not table_path.exists():
        print(f"Errore: file non trovato: {table_path}")
        sys.exit(1)

    # Elimina le celle precedenti
    if CELLS_DIR.exists():
        shutil.rmtree(CELLS_DIR)

    # 1. Estrai le celle usando il programma già esistente
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.classificatore.table_extractor",
            str(table_path),
        ],
        check=True,
    )

    # 2. Carica il modello già esistente
    model = joblib.load(MODEL_PATH)

    # 3. Recupera le celle
    cells = sorted(CELLS_DIR.glob("cell_*.png"))

    if not cells:
        print("Errore: nessuna cella trovata.")
        sys.exit(1)

    # 4. Classifica ogni cella
    results = {}

    for cell_path in cells:

        # cell_00_00.png -> riga 0, colonna 0
        parts = cell_path.stem.split("_")
        row = int(parts[1])
        col = int(parts[2])

        letter, confidence = predict_image(cell_path, model)

        results[(row, col)] = letter

        print(
            f"{cell_path.name}: "
            f"{letter} ({confidence * 100:.2f}%)"
        )

    # 5. Stampa la tabella finale
    max_row = max(row for row, col in results)
    max_col = max(col for row, col in results)

    print("\n=== TABELLA RICONOSCIUTA ===\n")

    for row in range(max_row + 1):
        line = []

        for col in range(max_col + 1):
            line.append(results[(row, col)])

        print(" ".join(line))


if __name__ == "__main__":
    main()