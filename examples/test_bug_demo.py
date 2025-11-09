#!/usr/bin/env python3
"""Demonstrate the bug in violation counting"""

import numpy as np
import jax.numpy as jnp

# Simulate a row: [5, 3, 2, 2, 7, 4, 8, 1, 9]
# Position 3 has value 2, and there's a duplicate 2 at position 2
# Correct violations for this row: 1 (the duplicate 2)

row = jnp.array([5, 3, 2, 2, 7, 4, 8, 1, 9])
i, j = 0, 3  # We're considering cell (0, 3) which currently has value 2

print("Row:", row)
print(f"Current value at position {j}: {row[j]}")
print(f"Actual violations in row: 1 (two 2's)\n")

# BUGGY CODE (current implementation)
print("=== BUGGY LOGIC ===")
for candidate_val in range(9):
    violations_buggy = 0.0
    row_count = jnp.sum(row == candidate_val)
    if row_count > 0:
        violations_buggy += row_count * 10.0
    if candidate_val == 1:  # value 2 in 0-indexed
        print(f"Candidate {candidate_val+1}: row_count={row_count}, "
              f"violations={violations_buggy/10:.0f} WRONG!")
        print(f"  Explanation: Counted current cell in row_count, so we think placing")
        print(f"  a 2 where a 2 already exists creates 2 violations instead of 1\n")

# CORRECT CODE
print("=== CORRECT LOGIC ===")
for candidate_val in range(9):
    violations_correct = 0.0
    # Exclude current cell when counting
    row_except_j = jnp.concatenate([row[:j], row[j+1:]])
    count_elsewhere = jnp.sum(row_except_j == candidate_val)
    violations_correct += count_elsewhere * 10.0
    if candidate_val == 1:  # value 2 in 0-indexed
        print(f"Candidate {candidate_val+1}: count_elsewhere={count_elsewhere}, "
              f"violations={violations_correct/10:.0f} CORRECT!")
        print(f"  Explanation: Excluded current cell, so placing a 2 where there's")
        print(f"  already one other 2 correctly shows 1 violation\n")

print("\n=== WHY THIS BREAKS THE SOLVER ===")
print("The buggy logic makes it seem like:")
print("  • Keeping the same value is WORSE than it actually is")
print("  • The sampler avoids good moves and makes random bad moves")
print("  • Energy goes down but violations stay high (contradictory!)")
print("  • The solver can't learn because the probability distribution is wrong")
