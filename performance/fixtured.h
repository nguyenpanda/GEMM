#ifndef FIXTURED_H
#define FIXTURED_H

#include "gemm.h"
#include <benchmark/benchmark.h>

namespace elementwise_FloatArray_Distributed {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"

class UnaryBufferFixture : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      buffer = new float[state.range(0)];
    } else {
      buffer = nullptr;
    }
  }

  void TearDown(const ::benchmark::State &state) override {
    if (buffer)
      delete buffer;
  }

  float *buffer;
};

class BinaryFixture : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      lhs = new float[state.range(0)];
      rhs = new float[state.range(0)];
      out = new float[state.range(0)];
    } else {
      lhs = nullptr;
      rhs = nullptr;
      out = nullptr;
    }
  }

  void TearDown(const ::benchmark::State &state) override {
    if (lhs)
      delete lhs;
    if (rhs)
      delete rhs;
    if (out)
      delete out;
  }

  float *lhs;
  float *rhs;
  float *out;
};

} // namespace elementwise_FloatArray_Distributed

namespace elementwise_Buffer_Distributed {

class UnaryBufferFixture : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      buffer = new Buffer<float>(state.range(0));
    } else {
      buffer = nullptr;
    }
  }

  void TearDown(const ::benchmark::State &state) override {
    if (buffer)
      delete buffer;
  }

  Buffer<float> *buffer;
};

class BinaryFixture : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      lhs = new Buffer<float>(state.range(0));
      rhs = new Buffer<float>(state.range(0));
      out = new Buffer<float>(state.range(0));
    } else {
      lhs = nullptr;
      rhs = nullptr;
      out = nullptr;
    }
  }

  void TearDown(const ::benchmark::State &state) override {
    if (lhs)
      delete lhs;
    if (rhs)
      delete rhs;
    if (out)
      delete out;
  }

  Buffer<float> *lhs;
  Buffer<float> *rhs;
  Buffer<float> *out;
};

}; // namespace elementwise_Buffer_Distributed

namespace elementwise_SplittableMatrix_Distributed {

class UnaryBufferFixture : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      buffer = new Buffer<float>(state.range(0));
      matrix = new SplittableMatrix<float>(buffer, state.range(0));

      initialize_random = new InitializeRandom<float>(-1, 1);
      initialize_random->fill(*buffer);
    } else {
      buffer = nullptr;
      matrix = new SplittableMatrix<float>(buffer, state.range(0));
      initialize_random = nullptr;
    }
  }

  void TearDown(const ::benchmark::State &state) override {
    if (matrix)
      delete matrix;
    if (buffer)
      delete buffer;
    if (initialize_random)
      delete initialize_random;
  }

  Buffer<float> *buffer;
  SplittableMatrix<float> *matrix;
  InitializeRandom<float> *initialize_random;
};

class BinaryFixture : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      lhs_buffer = new Buffer<float>(state.range(0));
      rhs_buffer = new Buffer<float>(state.range(0));
      out_buffer = new Buffer<float>(state.range(0));

      initialize_random = new InitializeRandom<float>(-1, 1);
      initialize_random->fill(*lhs_buffer);
      initialize_random->fill(*rhs_buffer);

      initialize_const = new InitializeConstant<float>(0);
      initialize_const->fill(*out_buffer);

      lhs = new SplittableMatrix<float>(lhs_buffer, state.range(0));
      rhs = new SplittableMatrix<float>(rhs_buffer, state.range(0));
      out = new SplittableMatrix<float>(out_buffer, state.range(0));
    } else {
      lhs_buffer = nullptr;
      rhs_buffer = nullptr;
      out_buffer = nullptr;

      // Create dummy SplittableMatrix objects on workers so they can be passed
      // by reference The actual data buffers are nullptr, but the
      // SplittableMatrix objects exist
      lhs = new SplittableMatrix<float>(lhs_buffer, state.range(0));
      rhs = new SplittableMatrix<float>(rhs_buffer, state.range(0));
      out = new SplittableMatrix<float>(out_buffer, state.range(0));

      initialize_random = nullptr;
      initialize_const = nullptr;
    }
  }

  void TearDown(const ::benchmark::State &state) override {
    if (lhs)
      delete lhs;
    if (rhs)
      delete rhs;
    if (out)
      delete out;

    if (lhs_buffer)
      delete lhs_buffer;
    if (rhs_buffer)
      delete rhs_buffer;
    if (out_buffer)
      delete out_buffer;

    if (initialize_random)
      delete initialize_random;
    if (initialize_const)
      delete initialize_const;
  }

  Buffer<float> *lhs_buffer, *rhs_buffer, *out_buffer;

  SplittableMatrix<float> *lhs, *rhs, *out;

  InitializeRandom<float> *initialize_random;
  InitializeConstant<float> *initialize_const;
};

#pragma clang diagnostic pop
}
; // namespace elementwise_SplittableMatrix_Distributed

#endif // FIXTURED_H
