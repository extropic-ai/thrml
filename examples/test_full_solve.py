#!/usr/bin/env python3
"""Test if fixed solver can fully solve Sudoku"""

import numpy as np
from sudoku_solver import SudokuSolver, create_example_puzzle

print("\n" + "="*70)
print("TESTING FIXED SUDOKU SOLVER - Aiming for Full Solution")
print("="*70)

# Create an easy puzzle
puzzle = create_example_puzzle("easy")

print("\nInitial puzzle:")
print(puzzle)
print(f"\nGiven clues: {np.count_nonzero(puzzle)}/81")

# Create solver
solver = SudokuSolver(puzzle, seed=123)  # Different seed

# Solve with MORE iterations and better annealing schedule
print("\nRunning with optimized parameters...")
print("(More iterations, slower cooling, lower final temp)\n")

solution, metrics = solver.solve_with_annealing(
    initial_temp=5.0,       # Start hotter for better exploration
    final_temp=0.01,        # Cool down more for exploitation
    cooling_rate=0.97,      # Slower cooling
    iterations_per_temp=50, # More iterations per temperature
    display_freq=50,        # Display less frequently
    display_delay=0.05      # Faster updates
)

# Verify solution
is_valid = solver.verify_solution(solution)

print("\n" + "="*70)
if is_valid:
    print("🎉 SUCCESS! FULL SOLUTION FOUND! 🎉")
else:
    row_v, col_v, box_v = solver.count_violations(solution)
    total_v = row_v + col_v + box_v
    print(f"Very close - only {total_v} violations remaining")
    print(f"  Row: {row_v}, Col: {col_v}, Box: {box_v}")

print("="*70)
