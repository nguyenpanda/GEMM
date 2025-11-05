include ./.env
DIR = $(CURDIR)

# SIMD instruction
CXXFLAGS += -mavx512f -march=native

# SIMD Report
CXXFLAGS += -Rpass=vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
# CXXFLAGS += -fno-slp-vectorize -Rpass-missed=loop-vectorize

# Debug mode
# CXXFLAGS += -g 
# MACRO += -DDEBUG_MODE

# OpenMP
MACRO += -DOMP_ENABLE

include ../Makefile
