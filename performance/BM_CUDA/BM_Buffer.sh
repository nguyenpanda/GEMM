mkdir -p compile_flags

rm -rf build
make compile EXTRA_MACRO="-DMATMUL_ORDER=2" >> compile_flags/BM_Buffer.txt
make execute EXECUTE=build/BM_Buffer.exe OMP_NUM_THREADS=8

