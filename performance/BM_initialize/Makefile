include ./.env
DIR = $(CURDIR)

# SIMD instruction
CXXFLAGS += -mavx512f -march=native -O3

# Debug mode
# CXXFLAGS += -g 
# MACRO += -DDEBUG_MODE

# OpenMP
MACRO += -DOMP_ENABLE

include ../Makefile
