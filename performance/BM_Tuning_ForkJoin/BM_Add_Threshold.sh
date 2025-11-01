rm -rf build
make compile EXTRA_CXXFLAGS="-O3" EXTRA_MACRO="-DMATMUL_ORDER=1"
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=8

rm -rf build
make compile EXTRA_CXXFLAGS="-O3" EXTRA_MACRO="-DMATMUL_ORDER=2"
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=8

rm -rf build
make compile EXTRA_CXXFLAGS="-O3" EXTRA_MACRO="-DMATMUL_ORDER=1 -DPRECISION_MODE"
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=8

rm -rf build
make compile EXTRA_CXXFLAGS="-O0" EXTRA_MACRO="-DMATMUL_ORDER=1"
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=8

rm -rf build
make compile EXTRA_CXXFLAGS="-O0" EXTRA_MACRO="-DMATMUL_ORDER=1"
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=4
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=6
make execute EXECUTE=build/BM_Add_Threshold.exe OMP_NUM_THREADS=8