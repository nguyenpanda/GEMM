#!/bin/bash

mkdir -p compile_flags

rm -rf build
make compile >> compile_flags/BM_MatMul_MPI_Baseline.txt

make execute EXECUTE=build/BM_MatMul_MPI_Baseline.exe MPI_NP=2
make execute EXECUTE=build/BM_MatMul_MPI_Baseline.exe MPI_NP=3
make execute EXECUTE=build/BM_MatMul_MPI_Baseline.exe MPI_NP=5
make execute EXECUTE=build/BM_MatMul_MPI_Baseline.exe MPI_NP=9
