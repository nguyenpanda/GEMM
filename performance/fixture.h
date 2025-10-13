#ifndef FIXTURE_H
#define FIXTURE_H

#include "gemm.h"
#include <benchmark/benchmark.h>

namespace elementwise_Buffer {

class UnaryBufferFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        buffer = new Buffer<float>(state.range(0));
    }

    void TearDown(const ::benchmark::State& state) override {
		delete buffer;
	}

	Buffer<float>* buffer;
};

class BinaryFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        lhs = new Buffer<float>(state.range(0));
        rhs = new Buffer<float>(state.range(0));
        out = new Buffer<float>(state.range(0));
    }

    void TearDown(const ::benchmark::State& state) override {
		delete lhs;
        delete rhs;
        delete out;
	}

    Buffer<float>* lhs;
    Buffer<float>* rhs;
    Buffer<float>* out;
};

}; // namespace elementwise

namespace elementwise_SplittableMatrix {

class UnaryBufferFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        buffer = new Buffer<float>(state.range(0));
        matrix = new SplittableMatrix<float>(buffer, state.range(0));
        
        initialize_random = new InitializeRandom<float>(-10, 10);
        initialize_random->fill(*buffer);
    }

    void TearDown(const ::benchmark::State& state) override {
		delete matrix;
        delete buffer;
        delete initialize_random;
	}

	Buffer<float>* buffer;
    SplittableMatrix<float>* matrix;
    InitializeRandom<float>* initialize_random;
};

class BinaryFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        lhs_buffer = new Buffer<float>(state.range(0));
        rhs_buffer = new Buffer<float>(state.range(0));
        out_buffer = new Buffer<float>(state.range(0));

        initialize_random = new InitializeRandom<float>(-10, 10);
        initialize_random->fill(*lhs_buffer);
        initialize_random->fill(*rhs_buffer);

        initialize_const = new InitializeConstant<float>(0);
        initialize_const->fill(*out_buffer);

        lhs = new SplittableMatrix<float>(lhs_buffer, state.range(0));
        rhs = new SplittableMatrix<float>(rhs_buffer, state.range(0));
        out = new SplittableMatrix<float>(out_buffer, state.range(0));
    }

    void TearDown(const ::benchmark::State& state) override {
        delete lhs;
        delete rhs;
        delete out;
        
        delete lhs_buffer;
        delete rhs_buffer;
        delete out_buffer;

        delete initialize_random;
        delete initialize_const;
	}

    Buffer<float> *lhs_buffer, *rhs_buffer, *out_buffer;

    SplittableMatrix<float> *lhs, *rhs, *out;

    InitializeRandom<float>* initialize_random;
    InitializeConstant<float>* initialize_const;
};

}; // namespace elementwise

#endif // FIXTURE_H
