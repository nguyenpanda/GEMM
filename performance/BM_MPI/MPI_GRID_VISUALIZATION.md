# MPI Grid Matrix Multiplication - Workload Distribution Visualization

## Overview
The MPI Grid algorithm distributes matrix multiplication across a **√P × √P grid** of processes, where P is the total number of MPI processes.

---

## Case 1: MPI_NP = 4 (2×2 Grid)

### Process Layout
```
┌─────────┬─────────┐
│ Rank 0  │ Rank 1  │
│ (0,0)   │ (0,1)   │
├─────────┼─────────┤
│ Rank 2  │ Rank 3  │
│ (1,0)   │ (1,1)   │
└─────────┴─────────┘

Grid Size: 2×2
```

### Matrix Partitioning (Example: 1024×1024 matrix)
Each process gets a **512×512** block:

**Matrix A:**
```
     0      512     1024
   ┌─────────┬─────────┐
 0 │  A₀₀    │  A₀₁    │
   │ (Rank0) │ (Rank1) │
512├─────────┼─────────┤
   │  A₁₀    │  A₁₁    │
   │ (Rank2) │ (Rank3) │
1024└─────────┴─────────┘
```

**Matrix B:**
```
     0      512     1024
   ┌─────────┬─────────┐
 0 │  B₀₀    │  B₀₁    │
   │ (Rank0) │ (Rank1) │
512├─────────┼─────────┤
   │  B₁₀    │  B₁₁    │
   │ (Rank2) │ (Rank3) │
1024└─────────┴─────────┘
```

### Work Distribution by Process 0 (Root)

**Step 1: Process 0 Creates Views**
- Creates 2×2 grid of zero-copy views for both A and B matrices
- No data copying - just pointer arithmetic!

**Step 2: Process 0 Sends to Workers (Ranks 1, 2, 3)**

**To Rank 1 (row=0, col=1):**
```
Send: A₀₀ + A₀₁  (k_block=0,1)
      B₀₁ + B₁₁  (k_block=0,1)
Tag:  0,1,2,3

Why: Rank 1 computes C₀₁ = A₀₀×B₀₁ + A₀₁×B₁₁
```

**To Rank 2 (row=1, col=0):**
```
Send: A₁₀ + A₁₁  (k_block=0,1)
      B₀₀ + B₁₀  (k_block=0,1)
Tag:  0,1,2,3

Why: Rank 2 computes C₁₀ = A₁₀×B₀₀ + A₁₁×B₁₀
```

**To Rank 3 (row=1, col=1):**
```
Send: A₁₀ + A₁₁  (k_block=0,1)
      B₀₁ + B₁₁  (k_block=0,1)
Tag:  0,1,2,3

Why: Rank 3 computes C₁₁ = A₁₀×B₀₁ + A₁₁×B₁₁
```

**Step 3: Process 0 Computes Its Own Work**
```
Local computation: C₀₀ = A₀₀×B₀₀ + A₀₁×B₁₀
Uses multiplyRecursive() with views (zero-copy)
```

**Step 4: Process 0 Receives Results**
```
From Rank 1 → C₀₁ block
From Rank 2 → C₁₀ block
From Rank 3 → C₁₁ block
```

### Worker Process Execution (Ranks 1, 2, 3)

Each worker:
1. **Receives blocks** for all k_block iterations (k=0 to 1)
2. **Accumulates result** into local buffer
3. **Sends back** final result to Rank 0

**Example - Rank 1's workflow:**
```
┌──────────────────────────────────────┐
│ Receive A₀₀, B₀₁  (k_block=0)        │
│ Compute: result = A₀₀ × B₀₁          │
├──────────────────────────────────────┤
│ Receive A₀₁, B₁₁  (k_block=1)        │
│ Compute: result += A₀₁ × B₁₁         │
├──────────────────────────────────────┤
│ Send result (C₀₁) back to Rank 0     │
└──────────────────────────────────────┘
```

---

## Case 2: MPI_NP = 9 (3×3 Grid)

### Process Layout
```
┌─────────┬─────────┬─────────┐
│ Rank 0  │ Rank 1  │ Rank 2  │
│ (0,0)   │ (0,1)   │ (0,2)   │
├─────────┼─────────┼─────────┤
│ Rank 3  │ Rank 4  │ Rank 5  │
│ (1,0)   │ (1,1)   │ (1,2)   │
├─────────┼─────────┼─────────┤
│ Rank 6  │ Rank 7  │ Rank 8  │
│ (2,0)   │ (2,1)   │ (2,2)   │
└─────────┴─────────┴─────────┘

Grid Size: 3×3
```

### Matrix Partitioning (Example: 1024×1024 matrix)
Each process gets approximately **341×341** block (1024/3 ≈ 341):

**Matrix A:**
```
     0      341     682     1024
   ┌─────────┬─────────┬──────┐
 0 │  A₀₀    │  A₀₁    │ A₀₂  │
   │ (Rank0) │ (Rank1) │(Rnk2)│
341├─────────┼─────────┼──────┤
   │  A₁₀    │  A₁₁    │ A₁₂  │
   │ (Rank3) │ (Rank4) │(Rnk5)│
682├─────────┼─────────┼──────┤
   │  A₂₀    │  A₂₁    │ A₂₂  │
   │ (Rank6) │ (Rank7) │(Rnk8)│
1024└─────────┴─────────┴──────┘
```

### Work Distribution by Process 0 (Root)

**Process 0 sends to 8 workers (Ranks 1-8):**

**To Rank 1 (row=0, col=1):**
```
k_block=0: Send A₀₀, B₀₁  (tag: 0, 1)
k_block=1: Send A₀₁, B₁₁  (tag: 2, 3)
k_block=2: Send A₀₂, B₂₁  (tag: 4, 5)

Rank 1 computes: C₀₁ = A₀₀×B₀₁ + A₀₁×B₁₁ + A₀₂×B₂₁
```

**To Rank 4 (row=1, col=1) - Center process:**
```
k_block=0: Send A₁₀, B₀₁  (tag: 0, 1)
k_block=1: Send A₁₁, B₁₁  (tag: 2, 3)
k_block=2: Send A₁₂, B₂₁  (tag: 4, 5)

Rank 4 computes: C₁₁ = A₁₀×B₀₁ + A₁₁×B₁₁ + A₁₂×B₂₁
```

**To Rank 8 (row=2, col=2) - Bottom-right corner:**
```
k_block=0: Send A₂₀, B₀₂  (tag: 0, 1)
k_block=1: Send A₂₁, B₁₂  (tag: 2, 3)
k_block=2: Send A₂₂, B₂₂  (tag: 4, 5)

Rank 8 computes: C₂₂ = A₂₀×B₀₂ + A₂₁×B₁₂ + A₂₂×B₂₂
```

**Process 0's Local Work:**
```
k_block=0: Compute A₀₀×B₀₀
k_block=1: Compute A₀₁×B₁₀
k_block=2: Compute A₀₂×B₂₀

Result: C₀₀ = A₀₀×B₀₀ + A₀₁×B₁₀ + A₀₂×B₂₀
```

### Full Communication Pattern

**Total sends by Rank 0:**
- 8 workers × 3 k_blocks × 2 matrices (A, B) = **48 messages**

**Message tags pattern:**
```
For each worker:
  k_block=0: tags 0 (A), 1 (B)
  k_block=1: tags 2 (A), 3 (B)
  k_block=2: tags 4 (A), 5 (B)
```

---

## Key Algorithm Details

### Zero-Copy Communication
```cpp
// Process 0 creates MPI derived datatype
MPI_Datatype A_subarray;
int A_sizes[2]    = {1024, 1024};  // Full matrix
int A_subsizes[2] = {341, 341};     // Block size
int A_starts[2]   = {0, 341};       // Starting position

MPI_Type_create_subarray(2, A_sizes, A_subsizes, A_starts, 
                         MPI_ORDER_C, MPI_FLOAT, &A_subarray);

// Send non-contiguous block directly (zero-copy!)
MPI_Isend(A.root->data, 1, A_subarray, dest_rank, tag, MPI_COMM_WORLD, &req);
```

### Worker Process Loop
```cpp
for (int k_block = 0; k_block < grid_size; ++k_block) {
    // Receive A block
    MPI_Recv(A_buffer->data, a_rows*a_cols, MPI_FLOAT, 0, k_block*2, ...);
    
    // Receive B block
    MPI_Recv(B_buffer->data, b_rows*b_cols, MPI_FLOAT, 0, k_block*2+1, ...);
    
    // Wrap in SplittableMatrix and compute
    SplittableMatrix<T> A_mat(A_buffer, a_rows, a_cols);
    SplittableMatrix<T> B_mat(B_buffer, b_rows, b_cols);
    
    // Accumulate result (uses recursive divide-and-conquer)
    multiplyRecursive(result, A_mat, B_mat);
}

// Send final result back to Rank 0
MPI_Send(result.root->data, my_rows*my_cols, MPI_FLOAT, 0, 0, ...);
```

---

## Performance Characteristics

### Communication Volume per Process

**For N×N matrix with P processes (grid_size = √P):**

- **Block size**: N/√P × N/√P
- **Messages per worker**: √P iterations × 2 (A and B)
- **Data per worker**: 2 × (N/√P)² × √P = 2N²/P elements

**Example (1024×1024):**

| Processes | Grid | Block Size | Messages/Worker | Data/Worker  |
|-----------|------|------------|-----------------|--------------|
| 4         | 2×2  | 512×512    | 4               | ~524K floats |
| 9         | 3×3  | 341×341    | 6               | ~233K floats |
| 16        | 4×4  | 256×256    | 8               | ~131K floats |

### Computation Distribution

**Each process computes**: (N/√P) × (N/√P) × N = N³/P operations

✅ **Perfect load balance** - all processes do equal work!

---

## Timeline Diagram (MPI_NP=4, simplified)

```
Time →
────────────────────────────────────────────────────────────

Rank 0: [Send to 1,2,3] [Compute C₀₀] [Recv from 1,2,3] [Done]
Rank 1: [Wait] [Recv A₀₀,B₀₁] [Recv A₀₁,B₁₁] [Compute C₀₁] [Send C₀₁]
Rank 2: [Wait] [Recv A₁₀,B₀₀] [Recv A₁₁,B₁₀] [Compute C₁₀] [Send C₁₀]
Rank 3: [Wait] [Recv A₁₀,B₀₁] [Recv A₁₁,B₁₁] [Compute C₁₁] [Send C₁₁]
```

**Key observation**: Rank 0 uses **non-blocking sends** (MPI_Isend), so it can start computing immediately!

---

## Summary

### MPI_NP = 4
- **Grid**: 2×2
- **Each process**: 1/4 of matrix
- **k_block iterations**: 2
- **Total messages**: 12 (3 workers × 2 k_blocks × 2 matrices)

### MPI_NP = 9
- **Grid**: 3×3
- **Each process**: 1/9 of matrix
- **k_block iterations**: 3
- **Total messages**: 48 (8 workers × 3 k_blocks × 2 matrices)

### Why This Works
1. **Cannon's Algorithm principle**: Each process gets row blocks from A and column blocks from B
2. **Zero-copy**: MPI derived datatypes send non-contiguous blocks without copying
3. **Recursive computation**: Each block uses divide-and-conquer (THRESHOLD=64)
4. **Perfect scaling**: Communication and computation both scale as O(N²/P)
