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


def compare_heuristics(grid, start, goal):
    """Confronta A* usando le due euristiche di SmartVacuum."""
    results = []

    for name, heuristic_name in (("A* con h", "h"), ("A* con h2", "h2")):
        problem = SmartVacuum(grid, start, goal)
        heuristic = getattr(problem, heuristic_name)

        start_time = time.perf_counter()
        node = astar_search(problem, h=heuristic)
        elapsed = time.perf_counter() - start_time

        result = {
            "name": name,
            "node": node,
            "nodes_expanded": problem.nodes_expanded,
            "time": elapsed,
            "cost": node.path_cost if node else None,
        }
        results.append(result)

        print(f"--- {name} ---")
        print(f"Nodi espansi: {result['nodes_expanded']}")
        print(f"Tempo: {result['time']:.4f} s")
        print(f"Costo: {result['cost']}")
        print()

    if all(result["node"] for result in results):
        assert results[0]["cost"] == results[1]["cost"]

    return results


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

    print("=== CONFRONTO TRA LE DUE EURISTICHE ===")
    compare_heuristics(grid, start, goal)
