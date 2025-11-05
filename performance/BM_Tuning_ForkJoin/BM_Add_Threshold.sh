mkdir -p compile_flags

rm -rf build
make compile >> compile_flags/BM_Add_Threshold.txt
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=2
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=8
