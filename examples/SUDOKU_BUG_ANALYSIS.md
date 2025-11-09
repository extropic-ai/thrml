# Sudoku Solver Bug Analysis & Fix

## Discovery

The user observed that the Sudoku solver wasn't learning to solve puzzles. Specifically:
- Violations **increased** from 32 → 44 over the annealing process
- Energy **decreased** (became more negative), creating a contradiction
- The final solutions showed interesting geometric patterns and symmetries
- The solver appeared to be sampling randomly rather than optimizing

## Root Cause: Off-by-One Violation Counting Bug

### The Bug

In the conditional probability computation for Gibbs sampling, the code counted violations **including the current cell** when it should have **excluded it**:

```python
# BUGGY CODE
row = grid[i, :]  # Includes cell at position j
row_count = jnp.sum(row == candidate_val)
violations += row_count * 10.0
```

### Why This Broke the Solver

When evaluating "What if I place `candidate_val` at cell `(i, j)`?", we need to know how many times that value **already appears elsewhere** in the row/column/box.

**Example:** Row `[5, 3, 2, 2, 7, 4, 8, 1, 9]` at position 3 (currently value 2)

Considering `candidate_val = 2`:
- **Buggy logic**: Counts 2 occurrences → claims 2 violations
- **Correct logic**: Counts 1 occurrence elsewhere → correctly identifies 1 violation
- **Impact**: The sampler thinks keeping the same value is worse than it actually is!

This creates a **distorted probability distribution** where:
1. Good moves (keeping valid values) appear bad
2. Bad moves (creating violations) appear equally bad
3. The sampler makes essentially random moves
4. Violations accumulate instead of decreasing
5. Energy decreases (because beta increases) but violations increase (contradictory!)

### The Geometric Patterns

The "interesting symmetries" the user observed were artifacts of random sampling with a broken energy function. The solver wasn't optimizing - it was drifting through state space with biased probabilities.

## The Fix

Exclude the current cell when counting violations:

```python
# FIXED CODE
# Row violations: exclude current position j
row = grid[i, :]
row_except_j = jnp.concatenate([row[:j], row[j+1:]])
count_in_row = jnp.sum(row_except_j == candidate_val)
violations += float(count_in_row) * 10.0

# Column violations: exclude current position i
col = grid[:, j]
col_except_i = jnp.concatenate([col[:i], col[i+1:]])
count_in_col = jnp.sum(col_except_i == candidate_val)
violations += float(count_in_col) * 10.0

# Box violations: exclude current cell
box_i, box_j = i % 3, j % 3
box_idx = box_i * 3 + box_j
box_except_current = jnp.concatenate([box_flat[:box_idx], box_flat[box_idx+1:]])
count_in_box = jnp.sum(box_except_current == candidate_val)
violations += float(count_in_box) * 10.0
```

## Results After Fix

### Before Fix
```
Initial violations: 32
Final violations: 44  (WORSE!)
Box violations: 13 → 10
Energy: decreasing (contradictory)
Behavior: Random drift, no learning
```

### After Fix
```
Initial violations: 37
Final violations: 4  (89% improvement!)
Box violations: 12 → 0  (PERFECT!)
Best achieved: 2 violations
Energy: correlates correctly with violations
Behavior: Clear optimization, learning happening
```

## Key Insights

1. **Energy-based sampling requires precise probability calculations**: A small bug in the energy function completely broke the learning dynamics.

2. **Contradiction detection**: When energy decreases but violations increase, that's a red flag for a bug in the energy calculation.

3. **Gibbs sampling subtlety**: When computing conditional probabilities for a cell, you must exclude that cell from the neighborhood counts.

4. **Geometric patterns**: Random sampling with biased probabilities can create visually interesting but meaningless patterns.

5. **Simulated annealing is robust**: Even with a severely broken energy function, the annealing framework still produced energy decrease (just optimizing the wrong thing).

## Remaining Challenges

The solver now works correctly but still faces challenges:
- Gets very close (2-4 violations) but doesn't always reach 0
- May need more iterations, better cooling schedule, or different sampling strategy
- Sudoku is a highly constrained problem where local minima can trap the solver

## Lessons Learned

1. Always test energy functions against ground truth
2. Watch for contradictions between metrics
3. Off-by-one errors in energy-based models cause silent failures
4. User observations (geometric patterns) provided crucial debugging hints
5. The bug was in probability calculation, not the annealing algorithm itself
