#!/bin/bash

filename=BM_MatMul_MPI_Baseline
    
mkdir -p compile_flags

rm -rf build
make compile >> compile_flags/$filename.txt

for NP in $(seq 9 32); do
    make execute EXECUTE=build/$filename.exe MPI_NP=$NP HOSTFILE=./hosts.txt
done
