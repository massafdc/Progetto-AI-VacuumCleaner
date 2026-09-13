from aima.search import Problem


class SmartVacuum(Problem):


    def __init__(self, grid, start, goal):

        self.grid_size = len(grid)

        if self.grid_size == 0:
            raise ValueError("La griglia non può essere vuota.")

        if any(len(row) != self.grid_size for row in grid):
            raise ValueError("La griglia deve essere quadrata.")

        self.goal_position = goal

        #ho aggiunto un contatore per sapere quanti stati esplora e quanto ha lavorato
        self.nodes_expanded = 0

        # Convertiamo la griglia in tuple per renderla immutabile
        # e quindi utilizzabile all'interno degli stati di ricerca.
        grid_tuple = tuple(tuple(row) for row in grid)

        initial_state = (start, grid_tuple)

        super().__init__(initial_state)

    def actions(self, state):
        """
        Restituisce le azioni applicabili nello stato corrente.
        """

        #il contatore si incrementa di 1 perchè ogni volta che si espande un nodo chiama il metodo action

        self.nodes_expanded += 1

        position, grid = state
        row, col = position

        possible_actions = []

        if self._is_valid_position(row - 1, col, grid):
            possible_actions.append("UP")

        if self._is_valid_position(row + 1, col, grid):
            possible_actions.append("DOWN")

        if self._is_valid_position(row, col - 1, grid):
            possible_actions.append("LEFT")

        if self._is_valid_position(row, col + 1, grid):
            possible_actions.append("RIGHT")

        current_cell = grid[row][col]

        if current_cell in ("D", "V"):
            possible_actions.append("CLEAN")

        return possible_actions

    def result(self, state, action):

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

    
    ## euristica

    def h(self, node):
        """
        Euristica A* basata su una Minimum Spanning Tree (MST).
        La stima considera:
        1. Il costo minimo necessario per pulire tutte le
        celle ancora sporche
        2. Il costo minimo necessario per collegare:
            - posizione attuale del robot
            - tutte le celle sporche
            - posizione finale
        utilizzando una Minimum Spanning Tree.
        Le distanze tra le celle sono calcolate con Manhattan.
        La distanza di Manhattan può sottostimare il vero costo
        di movimento in presenza di ostacoli X, quindi costituisce
        un lower bound.
        """

        position, grid = node.state

        # --------------------------------------------------------
        # 1. Raccolta delle celle ancora sporche
        # --------------------------------------------------------

        dirty_cells = []
        cleaning_cost = 0

        for r, grid_row in enumerate(grid):
            for c, cell in enumerate(grid_row):

                if cell == "D":
                    dirty_cells.append((r, c))
                    cleaning_cost += 1

                elif cell == "V":
                    dirty_cells.append((r, c))
                    cleaning_cost += 2

        # --------------------------------------------------------
        # 2. Se non ci sono più celle sporche
        # --------------------------------------------------------

        if not dirty_cells:
            row, col = position
            goal_row, goal_col = self.goal_position

            return (
                abs(row - goal_row)
                + abs(col - goal_col)
            )

        # --------------------------------------------------------
        # 3. Costruiamo l'insieme dei punti che devono essere
        #    collegati:
        #
        #    robot + celle sporche + goal
        # --------------------------------------------------------

        points = [position]

        points.extend(dirty_cells)

        points.append(self.goal_position)

        # --------------------------------------------------------
        # 4. Minimum Spanning Tree
        #
        #    Utilizziamo l'algoritmo di Prim.
        # --------------------------------------------------------

        visited = {0}

        mst_cost = 0

        while len(visited) < len(points):

            best_distance = float("inf")
            best_point = None

            # Cerchiamo il collegamento più economico
            # tra un punto già nella MST e uno ancora fuori.
            for i in visited:

                r1, c1 = points[i]

                for j in range(len(points)):

                    if j in visited:
                        continue

                    r2, c2 = points[j]

                    distance = (
                        abs(r1 - r2)
                        + abs(c1 - c2)
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_point = j

            # Aggiungiamo il collegamento minimo alla MST.
            mst_cost += best_distance

            visited.add(best_point)

        # --------------------------------------------------------
        # 5. Euristica finale
        #
        #    cleaning_cost:
        #        costo obbligatorio delle operazioni CLEAN
        #
        #    mst_cost:
        #        lower bound del movimento necessario per
        #        collegare robot, celle sporche e goal
        # --------------------------------------------------------

        return cleaning_cost + mst_cost

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
