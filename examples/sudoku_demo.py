#!/usr/bin/env python3
"""
Simplified Sudoku Solver Demo
Shows the annealing visualization and metrics in action
"""

import numpy as np
from sudoku_solver import SudokuSolver, create_example_puzzle

def main():
    print("\n" + "="*70)
    print("SUDOKU SOLVER DEMO - Simulated Annealing Visualization")
    print("="*70)
    print("\nThis demo shows:")
    print("  ✓ Real-time grid visualization")
    print("  ✓ Temperature/beta annealing schedule")
    print("  ✓ Energy tracking")
    print("  ✓ Constraint violation metrics (row/col/box)")
    print("  ✓ Convergence monitoring")
    print("="*70)

    # Create an easy puzzle
    puzzle = create_example_puzzle("easy")

    print("\nInitial puzzle:")
    print(puzzle)
    print(f"\nGiven clues: {np.count_nonzero(puzzle)}/81")

    # Create solver
    solver = SudokuSolver(puzzle, seed=42)

    # Solve with annealing - use fewer iterations for demo
    print("\nStarting annealing process...")
    print("(Running with reduced iterations for demonstration)\n")

    solution, metrics = solver.solve_with_annealing(
        initial_temp=3.0,
        final_temp=0.5,
        cooling_rate=0.92,
        iterations_per_temp=10,  # Reduced for speed
        display_freq=3,
        display_delay=0.15
    )

    # Verify solution
    is_valid = solver.verify_solution(solution)

    print("\n" + "="*70)
    if is_valid:
        print("SUCCESS! Found a valid solution ✓")
    else:
        row_v, col_v, box_v = solver.count_violations(solution)
        print(f"Partially solved - {row_v + col_v + box_v} violations remaining")
        print("(Increase iterations for full solution)")

    print("="*70)

    # Show final solution
    print("\nFinal grid:")
    grid = solution.reshape(9, 9)
    for i in range(9):
        if i % 3 == 0:
            print("  ┌───────┬───────┬───────┐" if i == 0 else "  ├───────┼───────┼───────┤")
        row_str = "  │ "
        for j in range(9):
            val = int(grid[i, j]) + 1
            is_given = puzzle[i, j] != 0
            if is_given:
                row_str += f"\033[94m{val}\033[0m "  # Blue for givens
            else:
                row_str += f"{val} "
            if (j + 1) % 3 == 0:
                row_str += "│ " if j < 8 else "│"
        print(row_str)
    print("  └───────┴───────┴───────┘")

    print("\nMetrics summary printed above shows:")
    print("  - Temperature schedule (hot → cold)")
    print("  - Energy evolution")
    print("  - Violation tracking per constraint type")
    print("  - Convergence detection")

    return solution, metrics


if __name__ == "__main__":
    main()
