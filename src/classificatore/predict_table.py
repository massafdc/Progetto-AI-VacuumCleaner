from pathlib import Path
import shutil
import sys
import time

import joblib

from aima.search import breadth_first_graph_search, astar_search
from .predict import predict_image, MODEL_PATH
from .table_extractor import extract_table_cells
from src.smart_vacuum import SmartVacuum


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
CELLS_DIR = BASE_DIR / "cells"


# ============================================================
# ESTRAZIONE E CLASSIFICAZIONE
# ============================================================

def extract_and_classify(table_path):
    """
    Estrae le celle dalla tabella e classifica ogni cella.

    Restituisce:
        {(riga, colonna): lettera}
    """

    # Elimina eventuali celle precedenti
    if CELLS_DIR.exists():
        shutil.rmtree(CELLS_DIR)

    # Percorso assoluto dell'immagine
    table_path = Path(table_path).resolve()

    # --------------------------------------------------------
    # ESTRAZIONE DELLA TABELLA
    # --------------------------------------------------------
    #
    # Prima table_extractor.py veniva eseguito come
    # processo separato tramite subprocess.
    #
    # Ora utilizziamo direttamente la funzione
    # extract_table_cells().
    #
    # --------------------------------------------------------

    extract_table_cells(
        table_path,
        output_dir=CELLS_DIR
    )

    # --------------------------------------------------------
    # CARICA IL MODELLO
    # --------------------------------------------------------

    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------------
    # RECUPERA LE CELLE ESTRATTE
    # --------------------------------------------------------

    cells = sorted(
        CELLS_DIR.glob("cell_*.png")
    )

    if not cells:
        raise RuntimeError(
            "Nessuna cella trovata dopo l'estrazione."
        )

    results = {}

    print("\n=== CLASSIFICAZIONE CELLE ===\n")

    for cell_path in cells:

        parts = cell_path.stem.split("_")

        row = int(parts[1])
        col = int(parts[2])

        letter, confidence = predict_image(
            cell_path,
            model
        )

        results[(row, col)] = letter

        print(
            f"{cell_path.name}: "
            f"{letter} "
            f"({confidence * 100:.2f}%)"
        )

    return results


# ============================================================
# CONVERSIONE IMMAGINE -> STATO SMART VACUUM
# ============================================================

def build_problem(results):
    """
    Converte la griglia classificata nello stato richiesto
    da SmartVacuum.

    S -> posizione iniziale del robot
    F -> posizione finale del robot

    Le celle S e F vengono considerate pulite (C).
    """

    if not results:
        raise ValueError(
            "La griglia classificata è vuota."
        )

    max_row = max(
        row for row, col in results
    )

    max_col = max(
        col for row, col in results
    )

    rows = max_row + 1
    cols = max_col + 1

    # La griglia deve essere quadrata
    if rows != cols:
        raise ValueError(
            f"La griglia deve essere quadrata: "
            f"trovata {rows}x{cols}."
        )

    grid = []

    start = None
    goal = None

    for row in range(rows):

        current_row = []

        for col in range(cols):

            if (row, col) not in results:
                raise ValueError(
                    f"Manca la cella ({row}, {col}) "
                    f"nella griglia."
                )

            letter = results[(row, col)]

            # ------------------------------------------------
            # POSIZIONE INIZIALE
            # ------------------------------------------------

            if letter == "S":

                if start is not None:
                    raise ValueError(
                        "La griglia contiene "
                        "più di una posizione S."
                    )

                start = (row, col)

                # S indica la posizione del robot,
                # quindi la cella viene considerata pulita.
                current_row.append("C")

            # ------------------------------------------------
            # POSIZIONE FINALE
            # ------------------------------------------------

            elif letter == "F":

                if goal is not None:
                    raise ValueError(
                        "La griglia contiene "
                        "più di una posizione F."
                    )

                goal = (row, col)

                # F indica la posizione finale,
                # quindi la cella viene considerata pulita.
                current_row.append("C")

            # ------------------------------------------------
            # CELLA NORMALE
            # ------------------------------------------------

            else:
                current_row.append(letter)

        grid.append(current_row)

    # Controlla che esistano S e F
    if start is None:
        raise ValueError(
            "Nessuna posizione S trovata nella griglia."
        )

    if goal is None:
        raise ValueError(
            "Nessuna posizione F trovata nella griglia."
        )

    return grid, start, goal


# ============================================================
# VISUALIZZAZIONE GRIGLIA
# ============================================================

def print_grid(grid):
    """Stampa una griglia in formato leggibile."""

    for row in grid:
        print(" ".join(row))


def print_recognized_grid(results):
    """Stampa la griglia così come è stata classificata."""

    max_row = max(
        row for row, col in results
    )

    max_col = max(
        col for row, col in results
    )

    print("\n=== TABELLA RICONOSCIUTA ===\n")

    for row in range(max_row + 1):

        line = []

        for col in range(max_col + 1):
            line.append(
                results[(row, col)]
            )

        print(" ".join(line))


# ============================================================
# RICERCA
# ============================================================

def run_search(
    name,
    search_fn,
    problem,
    **kwargs
):
    """
    Esegue un algoritmo di ricerca e raccoglie
    le statistiche.
    """

    start_time = time.perf_counter()

    node = search_fn(
        problem,
        **kwargs
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(f"\n--- {name} ---")

    print(
        f"Nodi espansi: "
        f"{problem.nodes_expanded}"
    )

    print(
        f"Tempo: "
        f"{elapsed:.6f} s"
    )

    if node:

        solution = node.solution()

        print(
            f"Lunghezza soluzione: "
            f"{len(solution)}"
        )

        print(
            f"Costo soluzione: "
            f"{node.path_cost}"
        )

        print("\nAzioni:")

        for i, action in enumerate(
            solution,
            start=1
        ):
            print(
                f"{i}. {action}"
            )

    else:

        solution = None

        print(
            "Nessuna soluzione trovata."
        )

    return {
        "node": node,
        "solution": solution,
        "nodes_expanded": problem.nodes_expanded,
        "time": elapsed,
        "cost": (
            node.path_cost
            if node
            else None
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # Controllo argomenti
    if len(sys.argv) != 3:

        print("Uso:")
        print(
            "python3 -m "
            "src.classificatore.predict_table "
            "percorso/tabella.png "
            "[1|2|3]"
        )

        print()
        print("Modalità:")
        print("  1 = BFS")
        print("  2 = A*")
        print("  3 = BFS + A*")

        sys.exit(1)

    table_path = Path(
        sys.argv[1]
    )

    search_mode = sys.argv[2]

    # Controllo modalità
    if search_mode not in ("1", "2", "3"):

        print(
            "\nErrore: modalità di ricerca non valida."
        )

        print("Usa:")
        print("  1 = BFS")
        print("  2 = A*")
        print("  3 = BFS + A*")

        sys.exit(1)

    # Controllo esistenza immagine
    if not table_path.exists():

        print(
            f"Errore: file non trovato: "
            f"{table_path}"
        )

        sys.exit(1)

    try:

        # ====================================================
        # 1. ESTRAZIONE + CLASSIFICAZIONE
        # ====================================================

        results = extract_and_classify(
            table_path
        )

        # ====================================================
        # 2. MOSTRA GRIGLIA RICONOSCIUTA
        # ====================================================

        print_recognized_grid(
            results
        )

        # ====================================================
        # 3. CONVERSIONE IN STATO SMART VACUUM
        # ====================================================

        grid, start, goal = build_problem(
            results
        )

        print(
            "\n=== STATO SMART VACUUM ===\n"
        )

        print(
            f"Start: {start}"
        )

        print(
            f"Goal:  {goal}"
        )

        print("\nGriglia:")

        print_grid(grid)

        # ====================================================
        # 4. BFS
        # ====================================================

        bfs_result = None

        if search_mode in ("1", "3"):

            problem_bfs = SmartVacuum(
                grid,
                start,
                goal
            )

            bfs_result = run_search(
                "BFS (ricerca non informata)",
                breadth_first_graph_search,
                problem_bfs
            )

        # ====================================================
        # 5. A*
        # ====================================================

        astar_result = None

        if search_mode in ("2", "3"):

            problem_astar = SmartVacuum(
                grid,
                start,
                goal
            )

            astar_result = run_search(
                "A* (ricerca informata)",
                astar_search,
                problem_astar,
                h=problem_astar.h
            )

        # ====================================================
        # 6. CONFRONTO
        # ====================================================

        if search_mode == "3":

            print(
                "\n=== CONFRONTO RICERCHE ===\n"
            )

            print(
                f"{'Algoritmo':<30}"
                f"{'Nodi':>10}"
                f"{'Tempo (s)':>15}"
                f"{'Costo':>10}"
            )

            print("-" * 65)

            print(
                f"{'BFS':<30}"
                f"{bfs_result['nodes_expanded']:>10}"
                f"{bfs_result['time']:>15.6f}"
                f"{str(bfs_result['cost']):>10}"
            )

            print(
                f"{'A*':<30}"
                f"{astar_result['nodes_expanded']:>10}"
                f"{astar_result['time']:>15.6f}"
                f"{str(astar_result['cost']):>10}"
            )

        print(
            "\n=== PIPELINE COMPLETATA ==="
        )

    except Exception as e:

        print(
            f"\nErrore: {e}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()