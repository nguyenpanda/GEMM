// MPI Matrix Multiplication Benchmark using Google Benchmark
// Grid Distribution with Fork-Join Pattern

#include <benchmark/benchmark.h>
#include <chrono>
#include <cmath>
#include <openmpi/mpi.h>
#include <vector>

// Matrix class from fj_mpi_grid.cpp
class Matrix {
private:
  int rows;
  int cols;
  double *data;
  int stride;
  bool owner;

public:
  static const int THRESHOLD = 64;

  Matrix(int rows, int cols = -1, bool zero = true)
      : rows(rows), cols(cols == -1 ? rows : cols),
        stride(cols == -1 ? rows : cols), owner(true) {
    data = new double[rows * this->cols];
    if (zero)
      std::fill(data, data + rows * this->cols, 0.0);
  }

  Matrix(double *base, int rows, int cols, int stride)
      : rows(rows), cols(cols), data(base), stride(stride), owner(false) {}

  Matrix(const Matrix &other)
      : rows(other.rows), cols(other.cols), stride(other.stride), owner(true) {
    data = new double[rows * cols];
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j)
        at(i, j) = other.at(i, j);
  }

  ~Matrix() {
    if (owner)
      delete[] data;
  }

  inline double &at(int i, int j) { return data[i * stride + j]; }
  inline double at(int i, int j) const { return data[i * stride + j]; }

  int size() const { return rows; }
  int getRows() const { return rows; }
  int getCols() const { return cols; }
  double *getData() { return data; }
  const double *getData() const { return data; }

  static Matrix constant(int n, double value) {
    Matrix M(n, -1, false);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        M.at(i, j) = value;
    return M;
  }

  Matrix subMatrix(int row, int col, int numRows, int numCols) const {
    return Matrix(data + row * stride + col, numRows, numCols, stride);
  }

  void extractBlock(int row, int col, int blockRows, int blockCols,
                    double *buffer) const {
    for (int i = 0; i < blockRows; ++i) {
      for (int j = 0; j < blockCols; ++j) {
        buffer[i * blockCols + j] = at(row + i, col + j);
      }
    }
  }

  void insertBlock(int row, int col, int blockRows, int blockCols,
                   const double *buffer) {
    for (int i = 0; i < blockRows; ++i) {
      for (int j = 0; j < blockCols; ++j) {
        at(row + i, col + j) = buffer[i * blockCols + j];
      }
    }
  }

  static void multiplyDirect(const Matrix &A, const Matrix &B, Matrix &C) {
    int m = A.rows;
    int k = A.cols;
    int n = B.cols;
    for (int i = 0; i < m; ++i)
      for (int kk = 0; kk < k; ++kk) {
        double aik = A.at(i, kk);
        for (int j = 0; j < n; ++j)
          C.at(i, j) += aik * B.at(kk, j);
      }
  }

  static void multiplyRecursive(const Matrix &A, const Matrix &B, Matrix &C) {
    int m = A.getRows();
    int k = A.getCols();
    int n = B.getCols();

    if (m <= THRESHOLD || k <= THRESHOLD || n <= THRESHOLD) {
      multiplyDirect(A, B, C);
      return;
    }

    int half_m = m / 2;
    int half_k = k / 2;
    int half_n = n / 2;
    int rem_m = m - half_m;
    int rem_k = k - half_k;
    int rem_n = n - half_n;

    Matrix A11 = A.subMatrix(0, 0, half_m, half_k);
    Matrix A12 = A.subMatrix(0, half_k, half_m, rem_k);
    Matrix A21 = A.subMatrix(half_m, 0, rem_m, half_k);
    Matrix A22 = A.subMatrix(half_m, half_k, rem_m, rem_k);

    Matrix B11 = B.subMatrix(0, 0, half_k, half_n);
    Matrix B12 = B.subMatrix(0, half_n, half_k, rem_n);
    Matrix B21 = B.subMatrix(half_k, 0, rem_k, half_n);
    Matrix B22 = B.subMatrix(half_k, half_n, rem_k, rem_n);

    Matrix C11 = C.subMatrix(0, 0, half_m, half_n);
    Matrix C12 = C.subMatrix(0, half_n, half_m, rem_n);
    Matrix C21 = C.subMatrix(half_m, 0, rem_m, half_n);
    Matrix C22 = C.subMatrix(half_m, half_n, rem_m, rem_n);

    Matrix temp11(half_m, half_n);
    multiplyRecursive(A11, B11, temp11);
    multiplyRecursive(A12, B21, C11);
    for (int i = 0; i < half_m; ++i)
      for (int j = 0; j < half_n; ++j)
        C11.at(i, j) += temp11.at(i, j);

    Matrix temp12(half_m, rem_n);
    multiplyRecursive(A11, B12, temp12);
    multiplyRecursive(A12, B22, C12);
    for (int i = 0; i < half_m; ++i)
      for (int j = 0; j < rem_n; ++j)
        C12.at(i, j) += temp12.at(i, j);

    Matrix temp21(rem_m, half_n);
    multiplyRecursive(A21, B11, temp21);
    multiplyRecursive(A22, B21, C21);
    for (int i = 0; i < rem_m; ++i)
      for (int j = 0; j < half_n; ++j)
        C21.at(i, j) += temp21.at(i, j);

    Matrix temp22(rem_m, rem_n);
    multiplyRecursive(A21, B12, temp22);
    multiplyRecursive(A22, B22, C22);
    for (int i = 0; i < rem_m; ++i)
      for (int j = 0; j < rem_n; ++j)
        C22.at(i, j) += temp22.at(i, j);
  }
};

namespace {

// The core MPI grid-based matrix multiplication work
void mpi_grid_matmul_work(int n) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Determine grid size based on number of processes
  int grid_size = (int)std::sqrt(size);
  int num_blocks = grid_size * grid_size;

  if (rank == 0) {
    // ROOT PROCESS: Create matrices and distribute grid blocks
    Matrix A_full = Matrix::constant(n, 5.0);
    Matrix B_full = Matrix::constant(n, 6.0);
    Matrix C(n);

    // Calculate block sizes for rows and columns
    int base_block_rows = n / grid_size;
    int extra_rows = n % grid_size;
    int base_block_cols = n / grid_size;
    int extra_cols = n % grid_size;

    // Store all data buffers and requests for asynchronous sending
    std::vector<std::vector<int>> all_sizes(num_blocks);
    std::vector<std::vector<double>> all_A_blocks(num_blocks);
    std::vector<std::vector<std::vector<double>>> all_B_blocks(num_blocks);
    std::vector<MPI_Request> send_requests;

    for (int block_id = 1; block_id < num_blocks; ++block_id) {
      int grid_row = block_id / grid_size;
      int grid_col = block_id % grid_size;

      int row_offset =
          grid_row * base_block_rows + std::min(grid_row, extra_rows);
      int block_rows = base_block_rows + (grid_row < extra_rows ? 1 : 0);

      int a_col_offset =
          grid_col * base_block_cols + std::min(grid_col, extra_cols);
      int a_block_cols = base_block_cols + (grid_col < extra_cols ? 1 : 0);

      all_A_blocks[block_id].resize(block_rows * a_block_cols);
      A_full.extractBlock(row_offset, a_col_offset, block_rows, a_block_cols,
                          all_A_blocks[block_id].data());

      all_B_blocks[block_id].resize(grid_size);
      std::vector<int> B_block_cols(grid_size);

      for (int b_col = 0; b_col < grid_size; ++b_col) {
        int b_col_offset =
            b_col * base_block_cols + std::min(b_col, extra_cols);
        int b_cols = base_block_cols + (b_col < extra_cols ? 1 : 0);
        B_block_cols[b_col] = b_cols;

        all_B_blocks[block_id][b_col].resize(a_block_cols * b_cols);
        B_full.extractBlock(a_col_offset, b_col_offset, a_block_cols, b_cols,
                            all_B_blocks[block_id][b_col].data());
      }

      all_sizes[block_id].resize(3 + grid_size);
      all_sizes[block_id][0] = block_rows;
      all_sizes[block_id][1] = a_block_cols;
      all_sizes[block_id][2] = grid_size;
      for (int i = 0; i < grid_size; ++i) {
        all_sizes[block_id][3 + i] = B_block_cols[i];
      }

      MPI_Request req;
      MPI_Isend(all_sizes[block_id].data(), 3 + grid_size, MPI_INT, block_id, 0,
                MPI_COMM_WORLD, &req);
      send_requests.push_back(req);

      MPI_Isend(all_A_blocks[block_id].data(), block_rows * a_block_cols,
                MPI_DOUBLE, block_id, 1, MPI_COMM_WORLD, &req);
      send_requests.push_back(req);

      for (int b_col = 0; b_col < grid_size; ++b_col) {
        MPI_Isend(all_B_blocks[block_id][b_col].data(),
                  a_block_cols * B_block_cols[b_col], MPI_DOUBLE, block_id,
                  2 + b_col, MPI_COMM_WORLD, &req);
        send_requests.push_back(req);
      }
    }

    // Process 0 computes its own block
    {
      int grid_row = 0;
      int grid_col = 0;

      int row_offset = 0;
      int block_rows = base_block_rows + (grid_row < extra_rows ? 1 : 0);
      int a_col_offset = 0;
      int a_block_cols = base_block_cols + (grid_col < extra_cols ? 1 : 0);

      Matrix A_block(block_rows, a_block_cols);
      A_full.extractBlock(row_offset, a_col_offset, block_rows, a_block_cols,
                          A_block.getData());

      for (int b_col = 0; b_col < grid_size; ++b_col) {
        int b_col_offset =
            b_col * base_block_cols + std::min(b_col, extra_cols);
        int b_cols = base_block_cols + (b_col < extra_cols ? 1 : 0);

        Matrix B_block(a_block_cols, b_cols);
        B_full.extractBlock(a_col_offset, b_col_offset, a_block_cols, b_cols,
                            B_block.getData());

        Matrix result_block(block_rows, b_cols);
        Matrix::multiplyRecursive(A_block, B_block, result_block);

        C.insertBlock(row_offset, b_col_offset, block_rows, b_cols,
                      result_block.getData());
      }
    }

    // JOIN phase: Receive results
    std::vector<std::vector<std::vector<double>>> all_result_bufs(num_blocks);
    std::vector<MPI_Request> recv_requests;

    for (int block_id = 1; block_id < num_blocks; ++block_id) {
      int grid_row = block_id / grid_size;
      int grid_col = block_id % grid_size;

      int row_offset =
          grid_row * base_block_rows + std::min(grid_row, extra_rows);
      int block_rows = base_block_rows + (grid_row < extra_rows ? 1 : 0);

      all_result_bufs[block_id].resize(grid_size);

      for (int res_col = 0; res_col < grid_size; ++res_col) {
        int col_offset =
            res_col * base_block_cols + std::min(res_col, extra_cols);
        int result_cols = base_block_cols + (res_col < extra_cols ? 1 : 0);

        all_result_bufs[block_id][res_col].resize(block_rows * result_cols);
        MPI_Request req;
        MPI_Irecv(all_result_bufs[block_id][res_col].data(),
                  block_rows * result_cols, MPI_DOUBLE, block_id, 3 + res_col,
                  MPI_COMM_WORLD, &req);
        recv_requests.push_back(req);
      }
    }

    MPI_Waitall(send_requests.size(), send_requests.data(),
                MPI_STATUSES_IGNORE);
    MPI_Waitall(recv_requests.size(), recv_requests.data(),
                MPI_STATUSES_IGNORE);

    // Accumulate results
    for (int block_id = 1; block_id < num_blocks; ++block_id) {
      int grid_row = block_id / grid_size;
      int grid_col = block_id % grid_size;

      int row_offset =
          grid_row * base_block_rows + std::min(grid_row, extra_rows);
      int block_rows = base_block_rows + (grid_row < extra_rows ? 1 : 0);

      for (int res_col = 0; res_col < grid_size; ++res_col) {
        int col_offset =
            res_col * base_block_cols + std::min(res_col, extra_cols);
        int result_cols = base_block_cols + (res_col < extra_cols ? 1 : 0);

        Matrix temp(block_rows, result_cols);
        std::copy(all_result_bufs[block_id][res_col].begin(),
                  all_result_bufs[block_id][res_col].end(), temp.getData());
        Matrix C_block =
            C.subMatrix(row_offset, col_offset, block_rows, result_cols);

        for (int i = 0; i < block_rows; ++i) {
          for (int j = 0; j < result_cols; ++j) {
            C_block.at(i, j) += temp.at(i, j);
          }
        }
      }
    }

  } else if (rank < num_blocks) {
    // WORKER PROCESS
    std::vector<int> sizes(3 + grid_size);
    MPI_Request size_req;
    MPI_Irecv(sizes.data(), 3 + grid_size, MPI_INT, 0, 0, MPI_COMM_WORLD,
              &size_req);
    MPI_Wait(&size_req, MPI_STATUS_IGNORE);

    int block_rows = sizes[0];
    int a_block_cols = sizes[1];
    int num_b_blocks = sizes[2];
    std::vector<int> B_block_cols(num_b_blocks);
    for (int i = 0; i < num_b_blocks; ++i) {
      B_block_cols[i] = sizes[3 + i];
    }

    std::vector<double> A_block_buf(block_rows * a_block_cols);
    MPI_Request a_req;
    MPI_Irecv(A_block_buf.data(), block_rows * a_block_cols, MPI_DOUBLE, 0, 1,
              MPI_COMM_WORLD, &a_req);

    std::vector<std::vector<double>> B_blocks(num_b_blocks);
    std::vector<MPI_Request> b_reqs(num_b_blocks);
    for (int b_col = 0; b_col < num_b_blocks; ++b_col) {
      B_blocks[b_col].resize(a_block_cols * B_block_cols[b_col]);
      MPI_Irecv(B_blocks[b_col].data(), a_block_cols * B_block_cols[b_col],
                MPI_DOUBLE, 0, 2 + b_col, MPI_COMM_WORLD, &b_reqs[b_col]);
    }

    MPI_Wait(&a_req, MPI_STATUS_IGNORE);
    MPI_Waitall(num_b_blocks, b_reqs.data(), MPI_STATUSES_IGNORE);

    Matrix A_block(block_rows, a_block_cols);
    std::copy(A_block_buf.begin(), A_block_buf.end(), A_block.getData());

    std::vector<Matrix> result_blocks;
    for (int b_col = 0; b_col < num_b_blocks; ++b_col) {
      Matrix B_block(a_block_cols, B_block_cols[b_col]);
      std::copy(B_blocks[b_col].begin(), B_blocks[b_col].end(),
                B_block.getData());

      Matrix result_block(block_rows, B_block_cols[b_col]);
      Matrix::multiplyRecursive(A_block, B_block, result_block);
      result_blocks.push_back(result_block);
    }

    std::vector<MPI_Request> send_reqs(num_b_blocks);
    for (int b_col = 0; b_col < num_b_blocks; ++b_col) {
      MPI_Isend(result_blocks[b_col].getData(),
                block_rows * B_block_cols[b_col], MPI_DOUBLE, 0, 3 + b_col,
                MPI_COMM_WORLD, &send_reqs[b_col]);
    }

    MPI_Waitall(num_b_blocks, send_reqs.data(), MPI_STATUSES_IGNORE);
  }
}

// The Google Benchmark function
void BM_MpiGridMatmul(benchmark::State &state) {
  double max_elapsed_second;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n = state.range(0); // matrix size

  for (auto _ : state) {
    // Barrier to ensure all ranks start together
    MPI_Barrier(MPI_COMM_WORLD);

    // Time the MPI operation locally
    auto start = std::chrono::high_resolution_clock::now();
    mpi_grid_matmul_work(n);
    auto end = std::chrono::high_resolution_clock::now();

    // Barrier to ensure all ranks finish together
    MPI_Barrier(MPI_COMM_WORLD);

    auto duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    double elapsed_seconds = duration.count();

    // Compute the maximum elapsed time across all ranks
    MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    // Only rank 0's time counts (UseManualTime will take this)
    state.SetIterationTime(max_elapsed_second);
  }

  // // Set additional metadata
  // if (rank == 0) {
  //   int size;
  //   MPI_Comm_size(MPI_COMM_WORLD, &size);
  //   int grid_size = (int)std::sqrt(size);
  //   state.counters["MatrixSize"] = n;
  //   state.counters["NumProcesses"] = size;
  //   state.counters["GridSize"] = grid_size;
  //   state.counters["GFLOPS"] = benchmark::Counter(
  //       2.0 * n * n * n / max_elapsed_second / 1e9,
  //       benchmark::Counter::kIsRate);
  // }
}

} // namespace

// Register the benchmark with different matrix sizes
BENCHMARK(BM_MpiGridMatmul)
    ->Arg(100)
    ->Arg(200)
    ->Arg(300)
    ->Arg(400)
    ->Arg(500)
    ->Arg(600)
    ->Arg(700)
    ->Arg(800)
    ->Arg(900)
    ->Arg(1000)
    ->Arg(1100)
    ->Arg(1200)
    ->Arg(1300)
    ->Arg(1400)
    ->Arg(2000)
    ->Arg(3000)
    ->Arg(4000)
    ->Arg(5000)
    ->Arg(6000)
    ->Arg(7000)
    ->Arg(8000)
    ->Arg(9000)
    ->Arg(10000)

    ->UseManualTime()
    ->Unit(benchmark::kSecond);

// NullReporter disables output from non-root ranks
class NullReporter : public ::benchmark::BenchmarkReporter {
public:
  bool ReportContext(const Context &) override { return true; }
  void ReportRuns(const std::vector<Run> &) override {}
  void Finalize() override {}
};

// Main: initialize MPI and select the proper reporter per rank
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ::benchmark::Initialize(&argc, argv);

  if (rank == 0) {
    // Root rank prints the benchmark output
    ::benchmark::RunSpecifiedBenchmarks();
  } else {
    // Other ranks run benchmarks silently
    NullReporter null;
    ::benchmark::RunSpecifiedBenchmarks(&null);
  }

  MPI_Finalize();
  return 0;
}
