#!/bin/bash

mkdir -p compile_flags

rm -rf build
make compile >> compile_flags/BM_MatMul_Hybrid.txt

make execute EXECUTE=build/BM_MatMul_Hybrid_Grid.exe MPI_NP=1 OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_MatMul_Hybrid_Grid.exe MPI_NP=4 OMP_NUM_THREADS=2
make execute EXECUTE=build/BM_MatMul_Hybrid_Grid.exe MPI_NP=9 HOSTFILE=./hosts.txt OMP_NUM_THREADS=2
make execute EXECUTE=build/BM_MatMul_Hybrid_Grid.exe MPI_NP=16 HOSTFILE=./hosts.txt OMP_NUM_THREADS=2