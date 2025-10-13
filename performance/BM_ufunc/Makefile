include ./.env
DIR = $(CURDIR)
OMP_NUM_THREADS = 4
OMP_ENABLE = 1
# CXXFLAGS += -g -fsanitize=address -fno-omit-frame-pointer
# CXXFLAGS += -mavx
# CXXFLAGS +=  -mfma -march=native -ftree-vectorize -fopenmp -ftree-vectorizer-verbose=2
CXXFLAGS += -S -mavx

include ../Makefile
