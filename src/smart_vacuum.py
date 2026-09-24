from aima.search import Problem


class SmartVacuum(Problem):

    def __init__(self, grid, start, goal):

        self.grid_size = len(grid)

        if self.grid_size == 0:
            raise ValueError("La griglia non può essere vuota.")

        if any(len(row) != self.grid_size for row in grid):
            raise ValueError("La griglia deve essere quadrata.")

        # --------------------------------------------------------
        # INFORMAZIONI STATICHE DEL PROBLEMA
        # --------------------------------------------------------

        # La griglia non cambia durante la ricerca.
        self.grid = tuple(tuple(row) for row in grid)

        self.goal_position = goal

        if not self._is_valid_position(*start, self.grid):
            raise ValueError("La posizione iniziale non è valida.")

        if not self._is_valid_position(*goal, self.grid):
            raise ValueError("La posizione del goal non è valida.")

        # Posizioni delle celle che inizialmente devono essere pulite.
        # Queste posizioni non cambiano durante la ricerca.
        self.dirty_positions = []

        # Numero di CLEAN necessarie per ogni cella:
        # D -> 1
        # V -> 2
        for row in range(self.grid_size):
            for col in range(self.grid_size):

                cell = self.grid[row][col]

                if cell == "D":
                    self.dirty_positions.append((row, col))

                elif cell == "V":
                    self.dirty_positions.append((row, col))

        self.dirty_positions = tuple(self.dirty_positions)

        # Mappa posizione -> indice nella tupla dirty_positions.
        # Serve per trovare rapidamente la cella da pulire.
        self.dirty_index = {
            position: index
            for index, position in enumerate(self.dirty_positions)
        }

        # --------------------------------------------------------
        # STATO INIZIALE
        # --------------------------------------------------------

        # Per ogni cella sporca:
        #
        # D -> 1 CLEAN rimanente
        # V -> 2 CLEAN rimanenti
        #
        # Esempio:
        #
        # dirty_positions = ((0, 2), (0, 3), (0, 4))
        #
        # remaining_cleaning = (2, 1, 2)
        #
        # Il primo V necessita di 2 CLEAN,
        # il D di 1,
        # il secondo V di 2.

        remaining_cleaning = []

        for position in self.dirty_positions:

            row, col = position
            cell = self.grid[row][col]

            if cell == "D":
                remaining_cleaning.append(1)

            elif cell == "V":
                remaining_cleaning.append(2)

        remaining_cleaning = tuple(remaining_cleaning)

        # --------------------------------------------------------
        # STATO
        # --------------------------------------------------------
        #
        # (posizione_robot, stato_pulizia)
        #
        # La griglia NON fa parte dello stato.
        #

        initial_state = (
            start,
            remaining_cleaning
        )

        # Contatore utilizzato per analizzare la ricerca.
        self.nodes_expanded = 0

        super().__init__(initial_state)

    # ============================================================
    # ACTIONS
    # ============================================================

    def actions(self, state):
        """
        Restituisce le azioni applicabili nello stato corrente.

        Stato:
            (posizione_robot, remaining_cleaning)
        """

        self.nodes_expanded += 1

        position, remaining_cleaning = state

        row, col = position

        possible_actions = []

        # Movimento verso l'alto
        if self._is_valid_position(
            row - 1,
            col,
            self.grid
        ):
            possible_actions.append("UP")

        # Movimento verso il basso
        if self._is_valid_position(
            row + 1,
            col,
            self.grid
        ):
            possible_actions.append("DOWN")

        # Movimento verso sinistra
        if self._is_valid_position(
            row,
            col - 1,
            self.grid
        ):
            possible_actions.append("LEFT")

        # Movimento verso destra
        if self._is_valid_position(
            row,
            col + 1,
            self.grid
        ):
            possible_actions.append("RIGHT")

        # CLEAN è possibile solo se la cella corrente
        # necessita ancora di almeno una pulizia.
        dirty_idx = self.dirty_index.get(position)

        if (
            dirty_idx is not None
            and remaining_cleaning[dirty_idx] > 0
        ):
            possible_actions.append("CLEAN")

        return possible_actions

    # ============================================================
    # RESULT
    # ============================================================

    def result(self, state, action):
        """
        Restituisce il nuovo stato dopo aver eseguito action.
        """

        position, remaining_cleaning = state

        row, col = position

        new_position = position

        # Copia della situazione di pulizia.
        new_remaining = list(remaining_cleaning)

        # --------------------------------------------------------
        # MOVIMENTI
        # --------------------------------------------------------

        if action == "UP":

            new_position = (
                row - 1,
                col
            )

        elif action == "DOWN":

            new_position = (
                row + 1,
                col
            )

        elif action == "LEFT":

            new_position = (
                row,
                col - 1
            )

        elif action == "RIGHT":

            new_position = (
                row,
                col + 1
            )

        # --------------------------------------------------------
        # CLEAN
        # --------------------------------------------------------

        elif action == "CLEAN":

            dirty_idx = self.dirty_index.get(position)

            if dirty_idx is None:
                raise ValueError(
                    "Non è possibile eseguire CLEAN su questa cella."
                )

            if new_remaining[dirty_idx] <= 0:
                raise ValueError(
                    "La cella è già completamente pulita."
                )

            # D: 1 -> 0
            # V: 2 -> 1 -> 0
            new_remaining[dirty_idx] -= 1

        else:

            raise ValueError(
                f"Azione non valida: {action}"
            )

        return (
            new_position,
            tuple(new_remaining)
        )

    # ============================================================
    # GOAL TEST
    # ============================================================

    def goal_test(self, state):
        """
        Il goal è raggiunto quando:
        1. il robot si trova sulla posizione finale;
        2. tutte le celle sono completamente pulite.
        """

        position, remaining_cleaning = state

        if position != self.goal_position:
            return False

        return all(
            remaining == 0
            for remaining in remaining_cleaning
        )

    # ============================================================
    # PATH COST
    # ============================================================

    def path_cost(self, c, state1, action, state2):
        """
        Ogni azione ha costo 1.
        """

        return c + 1

    # ============================================================
    # EURISTICA 1 
    # ============================================================

    def h(self, node):
        """
        Euristica per A*.

        Tiene conto di:
        - CLEAN ancora necessarie;
        - distanza dalle celle sporche;
        - distanza dalle celle sporche al goal.

        Utilizza la distanza di Manhattan.

        Gli ostacoli non vengono ignorati dal problema:
        impediscono fisicamente i movimenti in actions().
        La Manhattan rimane una stima inferiore della distanza
        reale anche in presenza di ostacoli.
        """

        position, remaining_cleaning = node.state

        row, col = position

        # --------------------------------------------------------
        # COSTO DELLE PULIZIE RIMANENTI
        # --------------------------------------------------------

        cleaning_cost = sum(remaining_cleaning)

        # Se non rimane nulla da pulire,
        # bisogna solamente raggiungere il goal.
        if cleaning_cost == 0:

            return (
                abs(row - self.goal_position[0])
                + abs(col - self.goal_position[1])
            )

        # --------------------------------------------------------
        # PARTE DI MOVIMENTO
        # --------------------------------------------------------

        movement_lower_bound = 0

        for index, remaining in enumerate(remaining_cleaning):

            if remaining == 0:
                continue

            dirty_row, dirty_col = self.dirty_positions[index]

            # Robot -> cella sporca
            distance_to_dirty = (
                abs(row - dirty_row)
                + abs(col - dirty_col)
            )

            # Cella sporca -> goal
            distance_to_goal = (
                abs(dirty_row - self.goal_position[0])
                + abs(dirty_col - self.goal_position[1])
            )

            # Qualunque soluzione deve necessariamente:
            #
            # robot -> cella sporca -> ... -> goal
            #
            # per ogni cella che deve essere pulita.
            #
            # Prendiamo il massimo perché ogni soluzione
            # deve visitare tutte queste celle.

            lower_bound = (
                distance_to_dirty
                + distance_to_goal
            )

            movement_lower_bound = max(
                movement_lower_bound,
                lower_bound
            )

        return (
            cleaning_cost
            + movement_lower_bound
        )


    # ============================================================
    # EURISTICA 2
    # ============================================================

    def h2(self, node):
        """
        Euristica per A* che tiene conto degli ostacoli X.

        Utilizza la distanza minima reale sulla griglia,
        calcolata tramite BFS.

        Tiene conto di:
        - CLEAN ancora necessarie;
        - distanza reale robot -> celle sporche;
        - distanza reale celle sporche -> goal.
        """

        position, remaining_cleaning = node.state

        # Numero di CLEAN ancora necessarie
        cleaning_cost = sum(remaining_cleaning)

        # Se non ci sono più celle da pulire,
        # bisogna solo raggiungere il goal.
        if cleaning_cost == 0:

            distance = self._grid_distance(
                position,
                self.goal_position
            )

            return cleaning_cost + distance

        movement_lower_bound = 0

        for index, remaining in enumerate(remaining_cleaning):

            if remaining == 0:
                continue

            dirty_position = self.dirty_positions[index]

            # Distanza reale robot -> cella sporca
            distance_to_dirty = self._grid_distance(
                position,
                dirty_position
            )

            # Distanza reale cella sporca -> goal
            distance_to_goal = self._grid_distance(
                dirty_position,
                self.goal_position
            )

            lower_bound = (
                distance_to_dirty
                + distance_to_goal
            )

            movement_lower_bound = max(
                movement_lower_bound,
                lower_bound
            )

        return cleaning_cost + movement_lower_bound


    def _grid_distance(self, start, goal):
        """
        Calcola la distanza minima tra due celle della griglia
        usando BFS.

        Le celle X non possono essere attraversate.

        Restituisce:
            - distanza minima se il goal è raggiungibile;
            - infinito se il goal non è raggiungibile.
        """

        if start == goal:
            return 0

        queue = [(start, 0)]
        visited = {start}

        index = 0

        while index < len(queue):

            (row, col), distance = queue[index]
            index += 1

            neighbours = [
                (row - 1, col),  # UP
                (row + 1, col),  # DOWN
                (row, col - 1),  # LEFT
                (row, col + 1)   # RIGHT
            ]

            for next_position in neighbours:

                next_row, next_col = next_position

                if not self._is_valid_position(
                    next_row,
                    next_col,
                    self.grid
                ):
                    continue

                if next_position in visited:
                    continue

                if next_position == goal:
                    return distance + 1

                visited.add(next_position)

                queue.append(
                    (next_position, distance + 1)
                )

        return float("inf")


    # ============================================================
    # UTILITY
    # ============================================================

    @staticmethod
    def _is_valid_position(row, col, grid):
        """
        Controlla che la posizione:
        - sia dentro la griglia;
        - non sia un ostacolo X.
        """

        size = len(grid)

        if row < 0 or row >= size:
            return False

        if col < 0 or col >= size:
            return False

        if grid[row][col] == "X":
            return False

        return True

    # ============================================================
    # RICOSTRUZIONE DELLA GRIGLIA
    # ============================================================

    def get_grid(self, state):
        """
        Ricostruisce la griglia corrispondente allo stato.

        È utile solamente per visualizzare/debuggare lo stato,
        non viene utilizzata come parte dello stato di ricerca.
        """

        position, remaining_cleaning = state

        new_grid = [
            list(row)
            for row in self.grid
        ]

        for index, remaining in enumerate(remaining_cleaning):

            row, col = self.dirty_positions[index]

            if remaining == 2:
                new_grid[row][col] = "V"

            elif remaining == 1:
                new_grid[row][col] = "D"

            else:
                new_grid[row][col] = "C"

        return tuple(
            tuple(row)
            for row in new_grid
        )