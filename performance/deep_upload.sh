#!/bin/bash

# RUN Init
cd BM_initialize/
./BM_InitializeRandom.sh
#./BM_Main.sh
cd ..

# RUN ufunc
cd BM_ufunc/
#./BM_RawTest_Matmul.sh
#./BM_RawTest_VecAdd.sh
cd ..

# RUN MatAdd
cd BM_MatAdd/
#./BM_MatAdd/BM_MatAdd.sh
cd ..

# RUN BM_Tuning_ForkJoin
cd BM_Tuning_ForkJoin/
#./BM_Add_Threshold.sh
#./BM_MatMul_GoddamnnnAdd.sh
#./BM_MatMul_SameThreshold.sh
cd ..

# RUN OMP Strassen
cd BM_OMP_Strassen/
# ./BM_OMP_Strassen.sh
cd ..
