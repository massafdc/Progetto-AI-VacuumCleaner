from pathlib import Path
import shutil
import sys
import time
import multiprocessing

import joblib

from aima.search import breadth_first_graph_search, astar_search
from .predict import predict_image, MODEL_PATH
from .table_extractor import extract_table_cells
from src.smart_vacuum import SmartVacuum
from src.simulatore.simulatore import simulate, create_gif


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
CELLS_DIR = BASE_DIR / "cells"

# Tempo massimo concesso a ciascun algoritmo di ricerca.
# Può essere modificato facilmente.

SEARCH_TIMEOUT = 20.0


# ============================================================
# ESTRAZIONE E CLASSIFICAZIONE
# ============================================================

def extract_and_classify(table_path):
    """
    Estrae le celle dalla tabella e classifica ogni cella.

    Restituisce:
        {(riga, colonna): lettera}
    """

    if CELLS_DIR.exists():
        shutil.rmtree(CELLS_DIR)

    table_path = Path(table_path).resolve()

    extract_table_cells(
        table_path,
        output_dir=CELLS_DIR
    )

    model = joblib.load(MODEL_PATH)

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

            if letter == "S":

                if start is not None:
                    raise ValueError(
                        "La griglia contiene "
                        "più di una posizione S."
                    )

                start = (row, col)
                current_row.append("C")

            elif letter == "F":

                if goal is not None:
                    raise ValueError(
                        "La griglia contiene "
                        "più di una posizione F."
                    )

                goal = (row, col)
                current_row.append("C")

            else:
                current_row.append(letter)

        grid.append(current_row)

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
# WORKER DELLA RICERCA
# ============================================================

def _search_worker(
    search_fn,
    problem,
    kwargs,
    connection
):
    """
    Esegue l'algoritmo di ricerca in un processo separato.

    Questa funzione deve essere definita a livello globale
    per essere compatibile anche con Windows (spawn).
    """

    try:

        start_time = time.perf_counter()

        node = search_fn(
            problem,
            **kwargs
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        if node:

            solution = node.solution()
            cost = node.path_cost

        else:

            solution = None
            cost = None

        connection.send({
            "success": True,
            "solution": solution,
            "cost": cost,
            "nodes_expanded": problem.nodes_expanded,
            "time": elapsed,
        })

    except Exception as e:

        connection.send({
            "success": False,
            "error": str(e),
        })

    finally:

        connection.close()


# ============================================================
# RICERCA CON TIMEOUT
# ============================================================

def run_search(
    name,
    search_fn,
    problem,
    timeout=SEARCH_TIMEOUT,
    **kwargs
):
    """
    Esegue un algoritmo di ricerca in un processo separato.

    Se l'algoritmo termina entro 'timeout' secondi,
    restituisce il risultato.

    Se supera il timeout, il processo viene terminato.

    Questa funzione è indipendente dall'algoritmo utilizzato:
    può essere usata con BFS, A*, oppure altri algoritmi.
    """

    print(f"\n--- {name} ---")

    print(
        f"Timeout: {timeout:.2f} s"
    )

    parent_connection, child_connection = (
        multiprocessing.Pipe()
    )

    process = multiprocessing.Process(
        target=_search_worker,
        args=(
            search_fn,
            problem,
            kwargs,
            child_connection
        )
    )

    start_time = time.perf_counter()

    process.start()

    # Il processo principale aspetta al massimo
    # il timeout specificato.
    process.join(timeout)

    # --------------------------------------------------------
    # CASO 1: L'algoritmo è ancora in esecuzione
    # --------------------------------------------------------

    if process.is_alive():

        print(
            "Timeout raggiunto."
        )

        print(
            "Interrompo la ricerca..."
        )

        process.terminate()
        process.join()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        parent_connection.close()

        print(
            f"Tempo: {elapsed:.6f} s"
        )

        print(
            "Nessuna soluzione trovata "
            "(ricerca interrotta)."
        )

        return {
            "node": None,
            "solution": None,
            "nodes_expanded": None,
            "time": elapsed,
            "cost": None,
            "timeout": True,
        }

    # --------------------------------------------------------
    # CASO 2: L'algoritmo è terminato
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if parent_connection.poll():

        result = parent_connection.recv()

    else:

        result = {
            "success": False,
            "error": (
                "Il processo di ricerca è terminato "
                "senza restituire un risultato."
            ),
        }

    parent_connection.close()

    # --------------------------------------------------------
    # ERRORE DURANTE LA RICERCA
    # --------------------------------------------------------

    if not result["success"]:

        print(
            f"Errore durante la ricerca: "
            f"{result['error']}"
        )

        return {
            "node": None,
            "solution": None,
            "nodes_expanded": None,
            "time": elapsed,
            "cost": None,
            "timeout": False,
        }

    # --------------------------------------------------------
    # RISULTATO
    # --------------------------------------------------------

    solution = result["solution"]
    cost = result["cost"]
    nodes_expanded = result["nodes_expanded"]

    print(
        f"Nodi espansi: "
        f"{nodes_expanded}"
    )

    print(
        f"Tempo: "
        f"{elapsed:.6f} s"
    )

    if solution is not None:

        print(
            f"Lunghezza soluzione: "
            f"{len(solution)}"
        )

        print(
            f"Costo soluzione: "
            f"{cost}"
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

        print(
            "Nessuna soluzione trovata."
        )

    return {
        "node": None,
        "solution": solution,
        "nodes_expanded": nodes_expanded,
        "time": elapsed,
        "cost": cost,
        "timeout": False,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 3:

        print("Uso:")
        print(
            "python3 -m "
            "src.classificatore.predict_table "
            "percorso/tabella.png "
            "[1|2|3|4]"
        )

        print()
        print("Modalità:")
        print("  1 = BFS")
        print("  2 = A* con h")
        print("  3 = A* con h2")
        print("  4 = confronto BFS, A* con h e A* con h2")

        sys.exit(1)

    table_path = Path(
        sys.argv[1]
    )

    search_mode = sys.argv[2]

    if search_mode not in ("1", "2", "3", "4"):

        print(
            "\nErrore: modalità di ricerca non valida."
        )

        print("Usa:")
        print("  1 = BFS")
        print("  2 = A* con h")
        print("  3 = A* con h2")
        print("  4 = confronto BFS, A* con h e A* con h2")

        sys.exit(1)

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

        if search_mode in ("1", "4"):

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
        # 5. A* con h
        # ====================================================

        astar_result = None

        if search_mode in ("2", "4"):

            problem_astar = SmartVacuum(
                grid,
                start,
                goal
            )

            astar_result = run_search(
                "A* (euristica h)",
                astar_search,
                problem_astar,
                h=problem_astar.h
            )

        astar_h2_result = None

        if search_mode in ("3", "4"):

            problem_astar_h2 = SmartVacuum(
                grid,
                start,
                goal
            )

            astar_h2_result = run_search(
                "A* (euristica h2)",
                astar_search,
                problem_astar_h2,
                h=problem_astar_h2.h2
            )

        # ====================================================
        # 6. CONFRONTO
        # ====================================================

        if search_mode == "4":

            print(
                "\n=== CONFRONTO RICERCHE ED EURISTICHE ===\n"
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
                f"{str(bfs_result['nodes_expanded']):>10}"
                f"{bfs_result['time']:>15.6f}"
                f"{str(bfs_result['cost']):>10}"
            )

            print(
                f"{'A* con h':<30}"
                f"{str(astar_result['nodes_expanded']):>10}"
                f"{astar_result['time']:>15.6f}"
                f"{str(astar_result['cost']):>10}"
            )

            print(
                f"{'A* con h2':<30}"
                f"{str(astar_h2_result['nodes_expanded']):>10}"
                f"{astar_h2_result['time']:>15.6f}"
                f"{str(astar_h2_result['cost']):>10}"
            )

        # ====================================================
        # 7. SIMULAZIONE
        # ====================================================

        if (
            astar_result
            and astar_result["solution"] is not None
        ):

            simulation_problem = problem_astar
            simulation_solution = (
                astar_result["solution"]
            )

        elif (
            astar_h2_result
            and astar_h2_result["solution"] is not None
        ):

            simulation_problem = problem_astar_h2
            simulation_solution = (
                astar_h2_result["solution"]
            )

        elif (
            bfs_result
            and bfs_result["solution"] is not None
        ):

            simulation_problem = problem_bfs
            simulation_solution = (
                bfs_result["solution"]
            )

        else:

            simulation_problem = None
            simulation_solution = None

        if simulation_solution is not None:

            print(
                "\n=== SIMULAZIONE ==="
            )

            session_dir = simulate(
                simulation_problem,
                simulation_solution
            )

            create_gif(session_dir)

        else:

            print(
                "\nNessuna soluzione da simulare."
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

    # Necessario per multiprocessing su Windows.
    multiprocessing.freeze_support()

    main()