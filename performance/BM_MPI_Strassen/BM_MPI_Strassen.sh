#!/bin/bash
export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH   # cho chắc

mkdir -p compile_flags

rm -rf build
rm -rf log/*
make compile >> compile_flags/BM_MPI_Strassen.txt
# ssh head-03 "mkdir -p /root/dfs/QHuong/GEMM/performance/BM_MPI_Strassen/build"
# scp -r /GEMM/performance/BM_MPI_Strassen/build/ head-03:/root/dfs/QHuong/GEMM/performance/BM_MPI_Strassen/build/
# ssh head-03 "chmod +x /root/dfs/QHuong/GEMM/performance/BM_MPI_Strassen/build/BM_MPI_Strassen.exe"

OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=2
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=3
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=4
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=5 
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=6
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=7
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=8
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=9 HOSTFILE=./hosts.txt
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=10 HOSTFILE=./hosts.txt
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=11 HOSTFILE=./hosts.txt
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=12 HOSTFILE=./hosts.txt
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=13 HOSTFILE=./hosts.txt
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=14 HOSTFILE=./hosts.txt
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=15 HOSTFILE=./hosts.txt
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make execute EXECUTE=build/BM_MPI_Strassen.exe MPI_NP=16 HOSTFILE=./hosts.txt




