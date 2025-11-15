#!/bin/bash

basename=$(basename "$0")
    
mkdir -p compile_flags

rm -rf build
make compile >> compile_flags/BM_MatMul_MPI_Baseline.txt

for NP in $(seq 1 32); do
    make execute EXECUTE=build/$basename.exe MPI_NP=$NP
done
