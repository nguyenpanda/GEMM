# MPI-specific settings
DIR = $(CURDIR)

# Include environment variables
include ./.env

# Enable OpenMP
MACRO += -DOMP_ENABLE

MACRO += -DDEBUG_MODE
# Enable MPI
# MACRO += -DUSE_MPI

# # MPI flags (detected from mpicxx --showme)
# MPI_INCLUDE = -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi
# MPI_LIBS = -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi

# Add MPI includes and libs
# EXTRA_CXXFLAGS += $(MPI_INCLUDE)
# LDFLAGS += $(MPI_LIBS)

# Disable Auto-Vectorize
# EXTRA_CXXFLAGS += -fno-vectorize -fno-slp-vectorize

# Order
# MACRO += -DMATMUL_ORDER=2
# MACRO += -DPRECISION_MODE

# Allow SIMD instruction
EXTRA_CXXFLAGS += -mavx2 -march=native

# SIMD Report
EXTRA_CXXFLAGS += -Rpass=vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
# EXTRA_CXXFLAGS += -fno-slp-vectorize -Rpass-missed=loop-vectorize

# Include parent Makefile
include ../Makefile
