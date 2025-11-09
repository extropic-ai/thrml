#!/usr/bin/env python3
"""Quick test of Sudoku solver without visualization delays"""

import numpy as np
from sudoku_solver import SudokuSolver, create_example_puzzle

# Create puzzle
puzzle = create_example_puzzle("easy")

print("Testing Sudoku Solver...")
print("Initial puzzle:")
print(puzzle)

# Create solver
solver = SudokuSolver(puzzle, seed=42)

# Solve with annealing (faster settings for testing)
solution, metrics = solver.solve_with_annealing(
    initial_temp=5.0,
    final_temp=0.1,
    cooling_rate=0.90,
    iterations_per_temp=20,
    display_freq=100,  # Display less frequently
    display_delay=0.0  # No delay
)

# Verify solution
is_valid = solver.verify_solution(solution)
print(f"\n{'='*70}")
print(f"Solution is {'VALID ✓' if is_valid else 'INVALID ✗'}")
print(f"{'='*70}")
