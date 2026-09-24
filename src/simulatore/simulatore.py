from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "simulazione"

CELL_COLORS = {
    "C": "#ffffff",
    "D": "#f4d58d",
    "V": "#c76b3f",
    "X": "#333333",
}

ROBOT_COLOR = "#2b6cb0"
GOAL_COLOR = "#d4af37"  # oro


# ============================================================
# DISEGNO DI UN SINGOLO FRAME
# ============================================================

def draw_frame(grid, position, step, action_label, goal_position, start_position, output_path):
    """
    Disegna lo stato corrente della griglia (robot compreso)
    e lo salva come immagine.
    """

    size = len(grid)
    fig, ax = plt.subplots(figsize=(size + 1, size + 1))

    goal_row, goal_col = goal_position
    start_row, start_col = start_position

    for row in range(size):
        for col in range(size):
            value = grid[row][col]
            color = CELL_COLORS.get(value, "#ffffff")

            ax.add_patch(patches.Rectangle(
                (col, size - 1 - row), 1, 1,
                facecolor=color, edgecolor="black", linewidth=1.5,
            ))

            is_goal_cell = (row, col) == (goal_row, goal_col)
            is_start_cell = (row, col) == (start_row, start_col)

            if value == "X":
                # X bianca ben visibile sullo sfondo scuro
                ax.text(
                    col + 0.5, size - 1 - row + 0.5, "X",
                    ha="center", va="center", fontsize=16,
                    color="white", fontweight="bold",
                )

            elif is_goal_cell:
                # Scritta FINISH al posto della lettera, solo
                # nella cella obiettivo
                ax.text(
                    col + 0.5, size - 1 - row + 0.5, "FINISH",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="#1a7a1a",
                )

            elif is_start_cell:
                # Scritta START al posto della lettera, solo
                # nella cella di partenza del robot
                ax.text(
                    col + 0.5, size - 1 - row + 0.5, "START",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="#2b6cb0",
                )

            else:
                ax.text(
                    col + 0.5, size - 1 - row + 0.5, value,
                    ha="center", va="center", fontsize=16,
                )

    # --------------------------------------------------------
    # Bordo netto sulla cella goal
    # --------------------------------------------------------

    ax.add_patch(patches.Rectangle(
        (goal_col, size - 1 - goal_row), 1, 1,
        fill=False, edgecolor=GOAL_COLOR, linewidth=3.5,
    ))

    # --------------------------------------------------------
    # Robot: icona disegnata (vista dall'alto)
    # --------------------------------------------------------

    robot_row, robot_col = position
    cx = robot_col + 0.5
    cy = size - 1 - robot_row + 0.5

    ax.add_patch(patches.Circle(
        (cx, cy), 0.32,
        facecolor="#4a4a4a", edgecolor="black", linewidth=1.5, zorder=5,
    ))

    ax.add_patch(patches.Circle(
        (cx, cy), 0.32, fill=False,
        edgecolor=ROBOT_COLOR, linewidth=3, zorder=6,
    ))

    ax.add_patch(patches.Circle(
        (cx, cy + 0.05), 0.08,
        facecolor=ROBOT_COLOR, edgecolor="black", linewidth=1, zorder=7,
    ))

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    title = f"Passo {step}"
    if action_label:
        title += f" — azione: {action_label}"
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


# ============================================================
# SIMULAZIONE DELL'INTERO PIANO
# ============================================================

def simulate(problem, actions, output_dir=OUTPUT_DIR):
    """
    Esegue il piano passo per passo e salva un'immagine per
    ogni stato attraversato.

    Lo stato contiene:
        (position, remaining_cleaning)

    La griglia visualizzata viene ricostruita tramite
    problem.get_grid(state).
    """

    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # STATO INIZIALE
    # --------------------------------------------------------

    state = problem.initial

    position, remaining_cleaning = state

    # Ricostruisce la griglia da visualizzare
    grid = problem.get_grid(state)

    # La posizione iniziale resta fissa per la visualizzazione
    start_position = position

    draw_frame(
        grid,
        position,
        step=0,
        action_label=None,
        goal_position=problem.goal_position,
        start_position=start_position,
        output_path=output_dir / "frame_00.png",
    )

    # --------------------------------------------------------
    # ESECUZIONE DELLE AZIONI
    # --------------------------------------------------------

    for index, action in enumerate(actions, start=1):

        state = problem.result(
            state,
            action
        )

        position, remaining_cleaning = state

        # Ricostruisce la griglia corrispondente
        # al nuovo stato
        grid = problem.get_grid(state)

        draw_frame(
            grid,
            position,
            step=index,
            action_label=action,
            goal_position=problem.goal_position,
            start_position=start_position,
            output_path=output_dir / f"frame_{index:02d}.png",
        )

    print(
        f"Simulazione salvata in: {output_dir}"
    )

    print(
        f"Frame generati: {len(actions) + 1}"
    )

    return output_dir


# ============================================================
# GIF ANIMATA
# ============================================================

def create_gif(output_dir=OUTPUT_DIR, gif_name="simulazione.gif", duration=800):
    from PIL import Image

    output_dir = Path(output_dir)
    frames = sorted(output_dir.glob("frame_*.png"))

    if not frames:
        raise FileNotFoundError(
            f"Nessun frame trovato in {output_dir}"
        )

    images = [Image.open(f) for f in frames]

    images[0].save(
        output_dir / gif_name,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )

    print(f"GIF salvata in: {output_dir / gif_name}")


# ============================================================
# TEST STANDALONE
# ============================================================

if __name__ == "__main__":
    from src.smart_vacuum import SmartVacuum
    from aima.search import astar_search

    grid = [
        ["D", "D", "C"],
        ["X", "V", "X"],
        ["C", "C", "C"],
    ]
    start = (2, 2)
    goal = (0, 2)

    problem = SmartVacuum(grid, start, goal)
    node = astar_search(problem, h=problem.h)

    session_dir = simulate(problem, node.solution())
    create_gif(session_dir)