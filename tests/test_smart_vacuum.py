from src.smart_vacuum import SmartVacuum

grid = [
["C", "D", "C"],
["X", "V", "C"],
["C", "D", "C"]
]

start = (0, 0)
goal = (2, 2)

problem = SmartVacuum(grid, start, goal)

state = problem.initial

state = problem.result(state, "RIGHT")
state = problem.result(state, "DOWN")
state = problem.result(state, "CLEAN")

print("\nDopo RIGHT + DOWN + CLEAN:")
print(state)

state = problem.result(state, "CLEAN")

print("\nDopo il secondo CLEAN:")
print(state)

print("\nTest goal_test:")

print("Goal sullo stato iniziale:")
print(problem.goal_test(problem.initial))

solved_grid = [
    ["C", "C", "C"],
    ["X", "C", "C"],
    ["C", "C", "C"]
]

solved_state = ((2, 2), tuple(tuple(row) for row in solved_grid))

print("\nGoal su uno stato risolto:")
print(problem.goal_test(solved_state))