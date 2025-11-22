#!/bin/bash

# mkdir -p compile_flags

rm -rf build
make compile

make execute EXECUTE=build/BM_MatMul_Hybrid_Baseline.exe MPI_NP=2 HOSTFILE=./hosts_grid.txt OMP_NUM_THREADS=8

# make execute EXECUTE=build/BM_MatMul_Hybrid_Grid.exe MPI_NP=4 HOSTFILE=./hosts_grid.txt OMP_NUM_THREADS=8

