#!/bin/bash

mkdir -p compile_flags

rm -rf build
make compile >> compile_flags/BM_MatMul_MPI_Grid.txt

make execute EXECUTE=build/BM_MatMul_MPI_Grid.exe MPI_NP=4
make execute EXECUTE=build/BM_MatMul_MPI_Grid.exe MPI_NP=9 HOSTFILE=./hosts.txt
make execute EXECUTE=build/BM_MatMul_MPI_Grid.exe MPI_NP=16 HOSTFILE=./hosts.txt

