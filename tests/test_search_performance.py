import time
from aima.search import breadth_first_graph_search, astar_search
from src.smart_vacuum import SmartVacuum


def run_search(name, search_fn, problem, **kwargs):
    start_time = time.perf_counter()
    node = search_fn(problem, **kwargs)
    elapsed = time.perf_counter() - start_time

    print(f"--- {name} ---")
    print(f"Nodi espansi: {problem.nodes_expanded}")
    print(f"Tempo: {elapsed:.4f} s")
    if node:
        print(f"Lunghezza soluzione: {len(node.solution())}")
    else:
        print("Nessuna soluzione trovata.")
    print()


if __name__ == "__main__":
    grid = [
        ["C", "D", "C"],
        ["X", "V", "C"],
        ["C", "D", "C"]
    ]
    start = (0, 0)
    goal = (2, 2)

    problem_bfs = SmartVacuum(grid, start, goal)
    run_search("BFS (non informata)", breadth_first_graph_search, problem_bfs)

    problem_astar = SmartVacuum(grid, start, goal)
    run_search("A* (informata, h)", astar_search, problem_astar, h=problem_astar.h)
