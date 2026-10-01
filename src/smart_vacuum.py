from aima.search import Problem


class SmartVacuum(Problem):
    def __init__(self, grid, start, goal):
        self.grid_size = len(grid)

        if self.grid_size == 0:
            raise ValueError("Errore, la griglia è vuota")
        if any(len(row) != self.grid_size for row in grid):
            raise ValueError("Errore, la griglia non è quadrata")

        self.goal_position = goal
        self.nodes_expanded = 0

        grid_tuple = tuple(tuple(row) for row in grid)
        initial_state = (start, grid_tuple)
        super().__init__(initial_state)

    def is_valid_position(self, row, col, grid):
        size = len(grid)
        if row< 0 or row >= size:
            return False
        if col< 0 or col >= size:
            return False
        if grid[row][col] == "X":
            return False
        return True

    def actions(self, state):
        self.nodes_expanded += 1
        position, grid = state
        row, col = position

        possible_actions = []
        if self.is_valid_position(row - 1, col, grid):
            possible_actions.append("UP")
        if self.is_valid_position(row + 1, col, grid):
            possible_actions.append("DOWN")
        if self.is_valid_position(row, col - 1, grid):
            possible_actions.append("LEFT")
        if self.is_valid_position(row, col + 1, grid):
            possible_actions.append("RIGHT")
        if grid[row][col] in ("D", "V"):
            possible_actions.append("CLEAN")

        return possible_actions

    def result(self, state, action):
        position, grid = state
        row, col = position

        new_grid = [list(grid_row) for grid_row in grid]
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

        return new_position, tuple(tuple(grid_row) for grid_row in new_grid)

    def goal_test(self, state):
        position, grid = state
        if position != self.goal_position:
            return False

        for grid_row in grid:
            for cell in grid_row:
                if cell in ("D", "V"):
                    return False
        return True

    def h(self, node):
        position, grid = node.state
        row, col = position

        cleaning_cost = 0
        movement_lower_bound = 0

        for grid_row in range(self.grid_size):
            for grid_col in range(self.grid_size):
                cell = grid[grid_row][grid_col]

                if cell == "D":
                    cleaning_cost += 1
                elif cell == "V":
                    cleaning_cost += 2
                else:
                    continue

                distance_to_dirty = abs(row - grid_row) + abs(col - grid_col)
                distance_to_goal = (
                    abs(grid_row - self.goal_position[0])
                    + abs(grid_col - self.goal_position[1])
                )
                lower_bound = distance_to_dirty + distance_to_goal
                movement_lower_bound = max(movement_lower_bound, lower_bound)

        if cleaning_cost == 0:
            return abs(row - self.goal_position[0]) + abs(col - self.goal_position[1])

        return cleaning_cost + movement_lower_bound

    def grid_distance(self, grid, start, goal):
        if start == goal:
            return 0

        queue = [(start, 0)]
        visited = {start}
        index = 0

        while index < len(queue):
            (row, col), distance = queue[index]
            index += 1

            for next_position in [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1),
            ]:
                if not self.is_valid_position(*next_position, grid):
                    continue
                if next_position in visited:
                    continue
                if next_position == goal:
                    return distance + 1

                visited.add(next_position)
                queue.append((next_position, distance + 1))

        return float("inf")

    def h2(self, node):
        position, grid = node.state

        cleaning_cost = 0
        movement_lower_bound = 0

        for grid_row in range(self.grid_size):
            for grid_col in range(self.grid_size):
                cell = grid[grid_row][grid_col]

                if cell == "D":
                    cleaning_cost += 1
                elif cell == "V":
                    cleaning_cost += 2
                else:
                    continue

                distance_to_dirty = self.grid_distance(grid, position, (grid_row, grid_col))
                distance_to_goal = self.grid_distance(grid, (grid_row, grid_col), self.goal_position)
                lower_bound = distance_to_dirty + distance_to_goal
                movement_lower_bound = max(movement_lower_bound, lower_bound)

        if cleaning_cost == 0:
            return self.grid_distance(grid, position, self.goal_position)

        return cleaning_cost + movement_lower_bound
