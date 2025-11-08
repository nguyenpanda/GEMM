mkdir -p compile_flags

rm -rf build
make compile EXTRA_MACRO="-DMATMUL_ORDER=1" >> compile_flags/BM_MatMul_GoddamnnnAdd.txt
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=2
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=3
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=5
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=7
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=8

rm -rf build
make compile EXTRA_MACRO="-DMATMUL_ORDER=2" >> compile_flags/BM_MatMul_GoddamnnnAdd.txt
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=2
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=3
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=5
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=7
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=8

rm -rf build
make compile EXTRA_MACRO="-DMATMUL_ORDER=1 -DPRECISION_MODE=1" >> compile_flags/BM_MatMul_GoddamnnnAdd.txt
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=2
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=3
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=5
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=7
make execute EXECUTE=build/BM_MatMul_GoddamnnnAdd.exe OMP_NUM_THREADS=8
