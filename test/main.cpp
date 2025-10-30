#include "main.h"
#include <filesystem>

typedef float DTYPE;

static const InitializeConstant<DTYPE> init_zero(0);
static const InitializeRandom<DTYPE> init_random(-1.0f, 1.0f);

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_local = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm_local, "%y-%m-%d_%H-%M-%S");
    return oss.str();
}

struct Args {
    std::string directory;
    bool sequence = false;
    std::string op;
};

Args parse_args(int argc, char** argv) {
    Args args;
    args.directory = "./buffer/" + get_timestamp();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--dir" && i + 1 < argc) {
            args.directory = argv[++i];
        } else if (arg == "--sequence") {
            args.sequence = true;
        } else if (arg == "--operator" && i + 1 < argc) {
            args.op = argv[++i];
        } else {
            fprintf(stderr, RED "Unknown or incomplete argument: %s\n" RESET, arg.c_str());
            exit(1);
        }
    }

    if (args.op.empty()) {
        fprintf(stderr, RED "Missing required argument: --operator (+, -, *, /, @)\n" RESET);
        exit(1);
    }
    if (args.op != "+" && args.op != "-" && args.op != "*" && args.op != "/" && args.op != "@") {
        fprintf(stderr, RED "Invalid operator: '%s'. Must be one of +, -, *, /, @\n" RESET, args.op.c_str());
        exit(1);
    }

    if (!args.directory.empty() && args.directory.back() != '/' && args.directory.back() != '\\')
        args.directory += '/';

    return args;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

	try {
        std::filesystem::path dir_path(args.directory);
        if (!std::filesystem::exists(dir_path)) {
            std::filesystem::create_directories(dir_path);
            printf(YELLOW "Created directory: %s\n" RESET, args.directory.c_str());
        }
    } catch (const std::filesystem::filesystem_error& e) {
        fprintf(stderr, RED "Failed to create directory '%s': %s\n" RESET,
                args.directory.c_str(), e.what());
        return 1;
    }

    {
        std::ofstream opfile(args.directory + "operator.txt");
        if (!opfile) {
            fprintf(stderr, RED "Failed to write operator file in %s\n" RESET, args.directory.c_str());
            return 1;
        }
        opfile << args.op;
        printf("Saved operator" GREEN " '%s'" RESET " to " GREEN "%soperator.txt\n" RESET,
               args.op.c_str(), args.directory.c_str());
    }

    printf(CYAN "===== INIT_METADATA =====\n" RESET);
	INIT_METADATA<DTYPE>();

    printf(CYAN "=========================\n" RESET);

	Buffer<DTYPE> buffer_lhs(MATRIX_RDIM, MATRIX_CDIM);
	Buffer<DTYPE> buffer_rhs(MATRIX_RDIM, MATRIX_CDIM);
	Buffer<DTYPE> buffer_out(MATRIX_RDIM, MATRIX_CDIM);

	init_random.fill(buffer_lhs);
    CODE_FOR_DEBUG_MODE(
        printf(GREEN "=== LHS ===\n" RESET);
        buffer_lhs.print();
    )

	init_random.fill(buffer_rhs);
    CODE_FOR_DEBUG_MODE(
        printf(GREEN "=== RHS ===\n" RESET);
        buffer_rhs.print();
    )

	init_zero.fill(buffer_out);
    CODE_FOR_DEBUG_MODE(
        printf(GREEN "=== OUT ===\n" RESET);
        buffer_out.print();
    )

	printf("Save " GREEN "`lhs` " RESET "buffer to " GREEN "%s\n" RESET, (args.directory + "lhs.nguyenpanda").c_str());
    SaveFile<DTYPE>::save(buffer_lhs, args.directory + "lhs.nguyenpanda");

    printf("Save " GREEN "`rhs` " RESET "buffer to " GREEN "%s\n" RESET, (args.directory + "rhs.nguyenpanda").c_str());
    SaveFile<DTYPE>::save(buffer_rhs, args.directory + "rhs.nguyenpanda");

	SplittableMatrix<DTYPE> out(&buffer_out, MATRIX_RDIM, MATRIX_CDIM);
	SplittableMatrix<DTYPE> lhs(&buffer_lhs, MATRIX_RDIM, MATRIX_CDIM);
	SplittableMatrix<DTYPE> rhs(&buffer_rhs, MATRIX_RDIM, MATRIX_CDIM);

	if (args.op == "+") {
        if (args.sequence) {
            printf(YELLOW "Running addition (sequential)...\n" RESET);
            ufunc::addition::Seq<DTYPE>::operate(out, lhs, rhs);
        } else {
            printf(YELLOW "Running addition (OpenMP)...\n" RESET);
            ufunc::addition::OmpForkJoin<DTYPE>::operate(out, lhs, rhs);
        }
    } else if (args.op == "@") {
        if (args.sequence) {
            printf(YELLOW "Running matrix multiplication (sequential)...\n" RESET);
            ufunc::matmul::Seq<DTYPE>::operate(out, lhs, rhs);
        } else {
            printf(YELLOW "Running matrix multiplication (OpenMP)...\n" RESET);
            ufunc::matmul::OmpForkJoin<DTYPE>::operate(out, lhs, rhs);
        }
    } else {
        fprintf(stderr, RED "Unsupported operator '%s' for execution.\n" RESET, args.op.c_str());
        return 1;
    }

	printf("Save " GREEN "`out` " RESET "buffer to " GREEN "%s\n" RESET, (args.directory + "out.nguyenpanda").c_str());
    SaveFile<DTYPE>::save(buffer_out, args.directory + "out.nguyenpanda");

    CODE_FOR_DEBUG_MODE(
        printf(GREEN "=== LHS ===\n" RESET);
        buffer_lhs.print();
    )

    CODE_FOR_DEBUG_MODE(
        printf(GREEN "=== RHS ===\n" RESET);
        buffer_rhs.print();
    )

    CODE_FOR_DEBUG_MODE(
        printf(GREEN "=== OUT ===\n" RESET);
        buffer_out.print();
    )

	return 0;
}
