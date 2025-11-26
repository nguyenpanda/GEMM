#!/bin/bash

mkdir -p compile_flags


rm -rf build
make compile >> compile_flags/BM_OMP_Strassen.txt

# make execute EXECUTE=build/BM_OMP_Strassen.exe OMP_NUM_THREADS=1
make execute EXECUTE=build/BM_OMP_Strassen.exe OMP_NUM_THREADS=2
make execute EXECUTE=build/BM_OMP_Strassen.exe OMP_NUM_THREADS=3
make execute EXECUTE=build/BM_OMP_Strassen.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_OMP_Strassen.exe OMP_NUM_THREADS=5
make execute EXECUTE=build/BM_OMP_Strassen.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_OMP_Strassen.exe OMP_NUM_THREADS=7
make execute EXECUTE=build/BM_OMP_Strassen.exe OMP_NUM_THREADS=8