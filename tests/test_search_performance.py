import time
import multiprocessing

from aima.search import breadth_first_graph_search, astar_search
from src.smart_vacuum import SmartVacuum


# ============================================================
# CONFIGURAZIONE
# ============================================================

SEARCH_TIMEOUT = 300.0


# ============================================================
# PROBLEMA
# ============================================================

grid = [
    ["C", "C", "X", "C"],
    ["V", "V", "C", "C"],
    ["D", "V", "V", "D"],
    ["D", "D", "X", "C"],
]

start = (0, 0)
goal = (3, 3)


# ============================================================
# WORKER
# ============================================================

def search_worker(search_function, problem, kwargs, connection):

    try:

        start_time = time.perf_counter()

        node = search_function(
            problem,
            **kwargs
        )

        elapsed = time.perf_counter() - start_time

        if node is not None:

            connection.send({
                "success": True,
                "time": elapsed,
                "nodes": problem.nodes_expanded,
                "cost": node.path_cost,
                "length": len(node.solution())
            })

        else:

            connection.send({
                "success": True,
                "time": elapsed,
                "nodes": problem.nodes_expanded,
                "cost": None,
                "length": None
            })

    except Exception as e:

        connection.send({
            "success": False,
            "error": str(e)
        })

    finally:

        connection.close()


# ============================================================
# RICERCA CON TIMEOUT
# ============================================================

def run_search(
    name,
    search_function,
    problem,
    **kwargs
):

    print(f"\n{name}")
    print("-" * 30)

    parent, child = multiprocessing.Pipe()

    process = multiprocessing.Process(
        target=search_worker,
        args=(
            search_function,
            problem,
            kwargs,
            child
        )
    )

    start_time = time.perf_counter()

    process.start()

    # Attende al massimo SEARCH_TIMEOUT secondi
    process.join(SEARCH_TIMEOUT)

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    if process.is_alive():

        process.terminate()
        process.join()

        elapsed = time.perf_counter() - start_time

        print("TIMEOUT")
        print(f"Tempo: {elapsed:.6f} s")

        parent.close()

        return

    # --------------------------------------------------------
    # RICERCA TERMINATA
    # --------------------------------------------------------

    elapsed = time.perf_counter() - start_time

    if parent.poll():

        result = parent.recv()

    else:

        print("Errore: nessun risultato ricevuto.")
        parent.close()
        return

    parent.close()

    if not result["success"]:

        print(f"Errore: {result['error']}")
        return

    print(f"Tempo:          {elapsed:.6f} s")
    print(f"Nodi espansi:   {result['nodes']}")
    print(f"Costo:          {result['cost']}")
    print(f"Lunghezza:      {result['length']}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=== GRIGLIA ===\n")

    for row in grid:
        print(" ".join(row))

    print(f"\nStart: {start}")
    print(f"Goal:  {goal}")

    print(f"\nTimeout per algoritmo: {SEARCH_TIMEOUT} s")

    # --------------------------------------------------------
    # BFS
    # --------------------------------------------------------

    problem_bfs = SmartVacuum(
        grid,
        start,
        goal
    )

    run_search(
        "BFS",
        breadth_first_graph_search,
        problem_bfs
    )

    # --------------------------------------------------------
    # A* con h
    # --------------------------------------------------------

    problem_h = SmartVacuum(
        grid,
        start,
        goal
    )

    run_search(
        "A* con h",
        astar_search,
        problem_h,
        h=problem_h.h
    )

    # --------------------------------------------------------
    # A* con h2
    # --------------------------------------------------------

    problem_h2 = SmartVacuum(
        grid,
        start,
        goal
    )

    run_search(
        "A* con h2",
        astar_search,
        problem_h2,
        h=problem_h2.h2
    )


if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()
