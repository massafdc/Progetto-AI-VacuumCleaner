from aima.search import Problem


class SmartVacuum(Problem):
    """
    Problema di ricerca per il dominio Smart Vacuum.

    La griglia è quadrata e contiene:
        C = Clean
        D = Dirty
        V = Very Dirty
        X = Inaccessible

    La posizione iniziale e quella finale vengono fornite
    separatamente rispetto alla griglia.
    """

    def __init__(self, grid, start, goal):
        """
        Parameters
        ----------
        grid : list[list[str]]
            Griglia quadrata contenente C, D, V, X.

        start : tuple[int, int]
            Posizione iniziale del robot (riga, colonna).

        goal : tuple[int, int]
            Posizione finale desiderata del robot (riga, colonna).
        """

        self.grid_size = len(grid)

        if self.grid_size == 0:
            raise ValueError("La griglia non può essere vuota.")

        if any(len(row) != self.grid_size for row in grid):
            raise ValueError("La griglia deve essere quadrata.")

        self.goal_position = goal

        # Convertiamo la griglia in tuple per renderla immutabile
        # e quindi utilizzabile all'interno degli stati di ricerca.
        grid_tuple = tuple(tuple(row) for row in grid)

        initial_state = (start, grid_tuple)

        super().__init__(initial_state)

    def actions(self, state):
        """
        Restituisce le azioni applicabili nello stato corrente.
        """

        position, grid = state
        row, col = position

        possible_actions = []

        # Movimento verso l'alto
        if self._is_valid_position(row - 1, col, grid):
            possible_actions.append("UP")

        # Movimento verso il basso
        if self._is_valid_position(row + 1, col, grid):
            possible_actions.append("DOWN")

        # Movimento verso sinistra
        if self._is_valid_position(row, col - 1, grid):
            possible_actions.append("LEFT")

        # Movimento verso destra
        if self._is_valid_position(row, col + 1, grid):
            possible_actions.append("RIGHT")

        # Pulizia della cella corrente
        current_cell = grid[row][col]

        if current_cell in ("D", "V"):
            possible_actions.append("CLEAN")

        return possible_actions

    def result(self, state, action):
        """
        Applica un'azione e restituisce il nuovo stato.
        """

        position, grid = state
        row, col = position

        new_grid = [list(r) for r in grid]
        new_position = position

        if action == "UP":
            new_position = (row - 1, col)

        elif action == "DOWN":
            new_position = (row + 1, col)

        elif action == "LEFT":
            new_position = (row, col - 1)

        elif action == "RIGHT":
            new_position = (row, col + 1)

        elif action == "CLEAN":

            if new_grid[row][col] == "D":
                new_grid[row][col] = "C"

            elif new_grid[row][col] == "V":
                new_grid[row][col] = "D"

        else:
            raise ValueError(f"Azione non valida: {action}")

        new_grid = tuple(tuple(r) for r in new_grid)

        return new_position, new_grid

    def goal_test(self, state):
        """
        Verifica se il goal è stato raggiunto.

        Il robot deve trovarsi nella posizione finale e
        tutte le celle accessibili devono essere pulite.
        """

        position, grid = state

        if position != self.goal_position:
            return False

        for row in grid:
            for cell in row:
                if cell in ("D", "V"):
                    return False

        return True

    def path_cost(self, c, state1, action, state2):
        """
        Ogni azione ha costo 1.
        """

        return c + 1

    
    # bella bro, qua definisco l'euristica

    def h(self, node):
    """
    Distanza di Manhattan dalla posizione corrente al goal
    + costo minimo di pulizia rimanente (1 per D, 2 per V).
    """
    position, grid = node.state
    row, col = position
    goal_row, goal_col = self.goal_position

    distance_to_goal = abs(row - goal_row) + abs(col - goal_col)

    cleaning_cost = sum(
        1 if cell == "D" else 2 if cell == "V" else 0
        for r in grid for cell in r
    )

    return distance_to_goal + cleaning_cost

    @staticmethod
    def _is_valid_position(row, col, grid):
        """
        Verifica se una posizione è interna alla griglia
        e non corrisponde a una cella X.
        """

        size = len(grid)

        if row < 0 or row >= size:
            return False

        if col < 0 or col >= size:
            return False

        if grid[row][col] == "X":
            return False

        return True
