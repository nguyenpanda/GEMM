rm -rf build
make compile EXTRA_MACRO="-DMATMUL_ORDER=1"
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=5
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=7
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=8

rm -rf build
make compile EXTRA_MACRO="-DMATMUL_ORDER=2"
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=5
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=7
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=8

rm -rf build
make compile EXTRA_MACRO="-DMATMUL_ORDER=1 -DPRECISION_MODE=1"
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=5
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=7
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=8
