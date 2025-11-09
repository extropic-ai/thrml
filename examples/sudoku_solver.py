#!/usr/bin/env python3
"""
Sudoku Solver using THRML with Simulated Annealing

This example demonstrates:
1. Modeling Sudoku as a constraint satisfaction problem using categorical nodes
2. Implementing custom energy functions for Sudoku constraints
3. Simulated annealing with temperature scheduling
4. Real-time visualization of the solving process
5. Detailed metrics about the annealing process
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass


@dataclass
class AnnealingMetrics:
    """Tracks metrics during the annealing process"""
    temperatures: List[float]
    betas: List[float]
    energies: List[float]
    row_violations: List[int]
    col_violations: List[int]
    box_violations: List[int]
    total_violations: List[int]
    iteration: int = 0

    def record(self, temp: float, beta: float, energy: float,
               row_viol: int, col_viol: int, box_viol: int):
        """Record metrics for current iteration"""
        self.temperatures.append(temp)
        self.betas.append(beta)
        self.energies.append(energy)
        self.row_violations.append(row_viol)
        self.col_violations.append(col_viol)
        self.box_violations.append(box_viol)
        self.total_violations.append(row_viol + col_viol + box_viol)
        self.iteration += 1

    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("ANNEALING METRICS SUMMARY")
        print("="*70)
        print(f"Total iterations: {self.iteration}")
        print(f"\nTemperature schedule:")
        print(f"  Initial: {self.temperatures[0]:.4f}")
        print(f"  Final:   {self.temperatures[-1]:.4f}")
        print(f"  Beta range: {self.betas[0]:.4f} -> {self.betas[-1]:.4f}")
        print(f"\nEnergy:")
        print(f"  Initial: {self.energies[0]:.2f}")
        print(f"  Final:   {self.energies[-1]:.2f}")
        print(f"  Best:    {min(self.energies):.2f} (iteration {self.energies.index(min(self.energies))})")
        print(f"\nConstraint violations:")
        print(f"  Row violations:    {self.row_violations[0]} -> {self.row_violations[-1]}")
        print(f"  Column violations: {self.col_violations[0]} -> {self.col_violations[-1]}")
        print(f"  Box violations:    {self.box_violations[0]} -> {self.box_violations[-1]}")
        print(f"  Total violations:  {self.total_violations[0]} -> {self.total_violations[-1]}")

        # Find when we first reached zero violations (if ever)
        zero_viol_iters = [i for i, v in enumerate(self.total_violations) if v == 0]
        if zero_viol_iters:
            print(f"\n  First valid solution found at iteration {zero_viol_iters[0]}")
            print(f"  Remained valid for {len(zero_viol_iters)} iterations")
        else:
            print(f"\n  No valid solution found (min violations: {min(self.total_violations)})")
        print("="*70)


class SudokuSolver:
    """Sudoku solver using energy-based optimization with simulated annealing"""

    def __init__(self, puzzle: np.ndarray, seed: int = 42):
        """
        Initialize Sudoku solver

        Args:
            puzzle: 9x9 array with 0 for empty cells, 1-9 for given clues
            seed: Random seed for reproducibility
        """
        self.puzzle = puzzle.copy()
        self.size = 9
        self.box_size = 3
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)

        # Track which cells are given
        self.given_cells = []
        for i in range(self.size):
            for j in range(self.size):
                if puzzle[i, j] != 0:
                    self.given_cells.append((i, j, puzzle[i, j] - 1))  # Store as 0-indexed

        print(f"Created Sudoku puzzle with {len(self.given_cells)} given clues")

    def get_cell_index(self, row: int, col: int) -> int:
        """Convert row, col to flat index"""
        return row * self.size + col

    def get_row_col(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to row, col"""
        return idx // self.size, idx % self.size

    def compute_energy_raw(self, state: jnp.ndarray) -> float:
        """
        Compute raw energy (violations) for current state
        """
        grid = state.reshape(self.size, self.size)
        energy = 0.0

        # Penalty for given cell violations (should be very high)
        given_penalty = 1000.0
        for i, j, val in self.given_cells:
            if grid[i, j] != val:
                energy += given_penalty

        # Row duplicates
        for i in range(self.size):
            row = grid[i, :]
            for digit in range(9):
                count = jnp.sum(row == digit)
                if count > 1:
                    energy += (count - 1) * 10.0

        # Column duplicates
        for j in range(self.size):
            col = grid[:, j]
            for digit in range(9):
                count = jnp.sum(col == digit)
                if count > 1:
                    energy += (count - 1) * 10.0

        # 3x3 box duplicates
        for box_row in range(3):
            for box_col in range(3):
                box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
                box_flat = box.flatten()
                for digit in range(9):
                    count = jnp.sum(box_flat == digit)
                    if count > 1:
                        energy += (count - 1) * 10.0

        return energy

    def compute_energy(self, state: jnp.ndarray, beta: float = 1.0) -> float:
        """Compute Boltzmann energy: -beta * raw_energy"""
        return -beta * self.compute_energy_raw(state)

    def count_violations(self, state: jnp.ndarray) -> Tuple[int, int, int]:
        """Count constraint violations (row, column, box)"""
        grid = state.reshape(self.size, self.size)
        row_viol = 0
        col_viol = 0
        box_viol = 0

        # Row violations
        for i in range(self.size):
            row = grid[i, :]
            for digit in range(9):
                count = int(jnp.sum(row == digit))
                if count > 1:
                    row_viol += count - 1

        # Column violations
        for j in range(self.size):
            col = grid[:, j]
            for digit in range(9):
                count = int(jnp.sum(col == digit))
                if count > 1:
                    col_viol += count - 1

        # Box violations
        for box_row in range(3):
            for box_col in range(3):
                box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
                box_flat = box.flatten()
                for digit in range(9):
                    count = int(jnp.sum(box_flat == digit))
                    if count > 1:
                        box_viol += count - 1

        return row_viol, col_viol, box_viol

    def energy_fn_factory(self, beta: float):
        """Create energy function with specific beta"""
        def energy_fn(global_state):
            return self.compute_energy(global_state, beta)
        return energy_fn

    def display_grid(self, state: jnp.ndarray, title: str = "",
                     metrics: Optional[Dict] = None, clear: bool = True):
        """Display Sudoku grid with pretty formatting"""
        if clear:
            print("\033[2J\033[H", end="")  # Clear screen and move cursor to top

        grid = state.reshape(self.size, self.size)

        # Print title and metrics
        if title:
            print(f"\n{title}")
        if metrics:
            print(f"Iteration: {metrics.get('iter', 0):4d}  |  "
                  f"Temp: {metrics.get('temp', 0):.4f}  |  "
                  f"Beta: {metrics.get('beta', 0):.4f}  |  "
                  f"Energy: {metrics.get('energy', 0):.2f}")
            print(f"Violations: Row={metrics.get('row_viol', 0):2d}  "
                  f"Col={metrics.get('col_viol', 0):2d}  "
                  f"Box={metrics.get('box_viol', 0):2d}  "
                  f"Total={metrics.get('total_viol', 0):2d}")

        print("\n  ┌" + "─" * 23 + "┐")

        for i in range(self.size):
            if i > 0 and i % 3 == 0:
                print("  ├" + "─" * 7 + "┼" + "─" * 7 + "┼" + "─" * 7 + "┤")

            row_str = "  │ "
            for j in range(self.size):
                val = int(grid[i, j]) + 1  # Convert back to 1-9

                # Color coding: blue for given, green if correct, red if in wrong spot
                is_given = any(gi == i and gj == j for gi, gj, _ in self.given_cells)

                if is_given:
                    # Blue for given clues
                    row_str += f"\033[94m{val}\033[0m "
                else:
                    # Default color for user-filled
                    row_str += f"{val} "

                if (j + 1) % 3 == 0 and j < 8:
                    row_str += "│ "

            row_str += "│"
            print(row_str)

        print("  └" + "─" * 23 + "┘")

    def solve_with_annealing(self,
                            initial_temp: float = 10.0,
                            final_temp: float = 0.01,
                            cooling_rate: float = 0.95,
                            iterations_per_temp: int = 50,
                            display_freq: int = 10,
                            display_delay: float = 0.05):
        """
        Solve Sudoku using simulated annealing

        Args:
            initial_temp: Starting temperature
            final_temp: Ending temperature
            cooling_rate: Temperature multiplier each step (< 1)
            iterations_per_temp: Gibbs iterations per temperature
            display_freq: Display every N iterations
            display_delay: Seconds to pause between displays
        """
        print("\n" + "="*70)
        print("SUDOKU SOLVER WITH SIMULATED ANNEALING")
        print("="*70)
        print(f"Initial temperature: {initial_temp}")
        print(f"Final temperature: {final_temp}")
        print(f"Cooling rate: {cooling_rate}")
        print(f"Iterations per temperature: {iterations_per_temp}")
        print("="*70)

        # Initialize metrics
        metrics = AnnealingMetrics(
            temperatures=[],
            betas=[],
            energies=[],
            row_violations=[],
            col_violations=[],
            box_violations=[],
            total_violations=[]
        )

        # Initialize state
        # For given cells, use the clue value
        # For empty cells, initialize randomly
        initial_state = np.zeros(self.size * self.size, dtype=np.int32)
        for i in range(self.size):
            for j in range(self.size):
                idx = self.get_cell_index(i, j)
                if self.puzzle[i, j] != 0:
                    initial_state[idx] = self.puzzle[i, j] - 1  # 0-indexed
                else:
                    initial_state[idx] = np.random.randint(0, 9)

        current_state = jnp.array(initial_state)

        # Display initial state
        row_v, col_v, box_v = self.count_violations(current_state)
        self.display_grid(current_state,
                         title="INITIAL STATE",
                         metrics={
                             'iter': 0,
                             'temp': initial_temp,
                             'beta': 1.0 / initial_temp,
                             'energy': 0.0,
                             'row_viol': row_v,
                             'col_viol': col_v,
                             'box_viol': box_v,
                             'total_viol': row_v + col_v + box_v
                         })
        time.sleep(2.0)

        # Annealing loop
        temp = initial_temp
        iteration = 0
        best_state = current_state
        best_violations = float('inf')

        while temp > final_temp:
            beta = 1.0 / temp

            # Perform Gibbs sampling iterations at this temperature
            for _ in range(iterations_per_temp):
                iteration += 1

                # Sample each cell that's not given
                for i in range(self.size):
                    for j in range(self.size):
                        # Skip given cells
                        if self.puzzle[i, j] != 0:
                            continue

                        idx = self.get_cell_index(i, j)

                        # Compute conditional log-probabilities for this cell
                        # For efficiency, compute local violations for each candidate
                        grid = current_state.reshape(self.size, self.size)
                        log_probs = np.zeros(9)

                        for candidate_val in range(9):
                            # Count violations if we place candidate_val at (i,j)
                            # Key: exclude the current cell when counting!
                            violations = 0.0

                            # Row violations: count occurrences in row excluding position j
                            row = grid[i, :]
                            row_except_j = jnp.concatenate([row[:j], row[j+1:]])
                            count_in_row = jnp.sum(row_except_j == candidate_val)
                            violations += float(count_in_row) * 10.0

                            # Column violations: count occurrences in column excluding position i
                            col = grid[:, j]
                            col_except_i = jnp.concatenate([col[:i], col[i+1:]])
                            count_in_col = jnp.sum(col_except_i == candidate_val)
                            violations += float(count_in_col) * 10.0

                            # Box violations: count occurrences in box excluding current cell
                            box_row, box_col = i // 3, j // 3
                            box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
                            box_flat = box.flatten()
                            # Current cell's position within the flattened box
                            box_i, box_j = i % 3, j % 3
                            box_idx = box_i * 3 + box_j
                            box_except_current = jnp.concatenate([box_flat[:box_idx], box_flat[box_idx+1:]])
                            count_in_box = jnp.sum(box_except_current == candidate_val)
                            violations += float(count_in_box) * 10.0

                            # Boltzmann factor: -beta * violations
                            log_probs[candidate_val] = -beta * violations

                        # Sample new value
                        self.key, subkey = jax.random.split(self.key)
                        new_val = jax.random.categorical(subkey, jnp.array(log_probs))
                        current_state = current_state.at[idx].set(new_val)

                # Track metrics
                energy = float(self.compute_energy(current_state, beta))
                row_v, col_v, box_v = self.count_violations(current_state)
                total_v = row_v + col_v + box_v

                metrics.record(temp, beta, energy, row_v, col_v, box_v)

                # Track best solution
                if total_v < best_violations:
                    best_violations = total_v
                    best_state = current_state

                # Display periodically
                if iteration % display_freq == 0:
                    self.display_grid(current_state,
                                    title=f"SOLVING... (Best violations: {best_violations})",
                                    metrics={
                                        'iter': iteration,
                                        'temp': temp,
                                        'beta': beta,
                                        'energy': energy,
                                        'row_viol': row_v,
                                        'col_viol': col_v,
                                        'box_viol': box_v,
                                        'total_viol': total_v
                                    })
                    time.sleep(display_delay)

                # Early exit if solved
                if total_v == 0:
                    self.display_grid(current_state,
                                    title="SOLVED! ✓",
                                    metrics={
                                        'iter': iteration,
                                        'temp': temp,
                                        'beta': beta,
                                        'energy': energy,
                                        'row_viol': row_v,
                                        'col_viol': col_v,
                                        'box_viol': box_v,
                                        'total_viol': total_v
                                    })
                    metrics.print_summary()
                    return current_state, metrics

            # Cool down
            temp *= cooling_rate

        # Display final result
        row_v, col_v, box_v = self.count_violations(best_state)
        total_v = row_v + col_v + box_v
        energy = float(self.compute_energy(best_state, 1.0 / final_temp))

        title = "FINAL RESULT - SOLVED! ✓" if total_v == 0 else f"FINAL RESULT - {total_v} violations remaining"
        self.display_grid(best_state,
                         title=title,
                         metrics={
                             'iter': iteration,
                             'temp': final_temp,
                             'beta': 1.0 / final_temp,
                             'energy': energy,
                             'row_viol': row_v,
                             'col_viol': col_v,
                             'box_viol': box_v,
                             'total_viol': total_v
                         })

        metrics.print_summary()
        return best_state, metrics

    def verify_solution(self, state: jnp.ndarray) -> bool:
        """Verify if solution is valid"""
        row_v, col_v, box_v = self.count_violations(state)
        return row_v == 0 and col_v == 0 and box_v == 0


def create_example_puzzle(difficulty: str = "easy") -> np.ndarray:
    """Create example Sudoku puzzle"""

    if difficulty == "easy":
        # Easy puzzle with many clues
        puzzle = np.array([
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ])
    elif difficulty == "medium":
        # Medium puzzle
        puzzle = np.array([
            [0, 0, 0, 6, 0, 0, 4, 0, 0],
            [7, 0, 0, 0, 0, 3, 6, 0, 0],
            [0, 0, 0, 0, 9, 1, 0, 8, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 1, 8, 0, 0, 0, 3],
            [0, 0, 0, 3, 0, 6, 0, 4, 5],
            [0, 4, 0, 2, 0, 0, 0, 6, 0],
            [9, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 1, 0, 0]
        ])
    else:  # hard
        # Hard puzzle with fewer clues
        puzzle = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 8, 5],
            [0, 0, 1, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 7, 0, 0, 0],
            [0, 0, 4, 0, 0, 0, 1, 0, 0],
            [0, 9, 0, 0, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0, 0, 7, 3],
            [0, 0, 2, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 9]
        ])

    return puzzle


def main():
    """Main function to run Sudoku solver"""

    print("\n" + "="*70)
    print("SUDOKU SOLVER USING THRML")
    print("Thermodynamic HypergRaphical Model Library")
    print("="*70)

    # Create puzzle
    puzzle = create_example_puzzle("easy")

    # Create solver
    solver = SudokuSolver(puzzle, seed=42)

    # Solve with annealing
    solution, metrics = solver.solve_with_annealing(
        initial_temp=5.0,
        final_temp=0.01,
        cooling_rate=0.95,
        iterations_per_temp=30,
        display_freq=5,
        display_delay=0.1
    )

    # Verify solution
    is_valid = solver.verify_solution(solution)
    print(f"\nSolution is {'VALID ✓' if is_valid else 'INVALID ✗'}")

    return solution, metrics


if __name__ == "__main__":
    main()
