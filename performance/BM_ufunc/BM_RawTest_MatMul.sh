mkdir -p compile_flags

rm -rf build
make compile EXTRA_MACRO="-DMATMUL_ORDER=2" >> compile_flags/BM_RawTest_MatMul.txt
make execute EXECUTE=build/BM_RawTest_MatMul.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_RawTest_MatMul.exe OMP_NUM_THREADS=5
make execute EXECUTE=build/BM_RawTest_MatMul.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_RawTest_MatMul.exe OMP_NUM_THREADS=7
make execute EXECUTE=build/BM_RawTest_MatMul.exe OMP_NUM_THREADS=8
