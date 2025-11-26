#!/bin/bash

mkdir -p compile_flags

rm -rf build
make compile >> compile_flags/BM_Hybrid_Strassen.txt

# OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_Hybrid_Strassen.exe MPI_NP=2 OMP_NUM_THREADS=4
# OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_Hybrid_Strassen.exe MPI_NP=4 OMP_NUM_THREADS=2
# OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_Hybrid_Strassen.exe MPI_NP=4 HOSTFILE=./hosts.txt OMP_NUM_THREADS=2
# OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_Hybrid_Strassen.exe MPI_NP=4 HOSTFILE=./hosts.txt OMP_NUM_THREADS=2
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_Hybrid_Strassen.exe MPI_NP=4 HOSTFILE=./hosts.txt OMP_NUM_THREADS=16