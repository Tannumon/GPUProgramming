#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>    // CPU timing

// -------------------------
// CUDA error-checking macro
// -------------------------
#define CUDA_CHECK(stmt)                                                     \
do {                                                                         \
    cudaError_t err = (stmt);                                                \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA ERROR %s (%d): %s at %s:%d\n",                 \
                #stmt, int(err), cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

#define CUBLAS_CHECK(stmt) do {                            \
    cublasStatus_t stat = (stmt);                          \
    if (stat != CUBLAS_STATUS_SUCCESS) {                   \
        fprintf(stderr, "%s failed\n", #stmt);             \
        std::exit(EXIT_FAILURE);                           \
    }                                                      \
} while (0)

// -------------------------
// ANSI colors
// -------------------------
#if defined(_WIN32)
  #define ANSI_GREEN ""
  #define ANSI_RED   ""
  #define ANSI_RESET ""
#else
  #define ANSI_GREEN "\x1b[32m"
  #define ANSI_RED   "\x1b[31m"
  #define ANSI_RESET "\x1b[0m"
#endif

// -----------------------------------------
// cuBLAS Implementation
// -----------------------------------------
float cublasVecAdd(const float* __restrict__ dA,
                   float* __restrict__ dB,
                   float* __restrict__ hOut,
                   int N,
                   float alpha)
{
    assert(N > 0);
    const size_t numBytes = static_cast<size_t>(N) * sizeof(float);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    // In-place: dB = alpha*dA + dB
    CUBLAS_CHECK(cublasSaxpy(handle, N, &alpha, dA, 1, dB, 1));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    CUDA_CHECK(cudaMemcpy(hOut, dB, numBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    return elapsedMs;
}

// -------------------------
// CPU reference (for check)
// -------------------------
void vecAddHost(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                double alpha)
{
    int N = int(A.size());
    for (int i = 0; i < N; ++i) {
        C[i] = alpha*A[i] + B[i];
    }
}

// -------------------------
// Helper to init input data
// -------------------------
void initRandom(std::vector<float>& v, uint64_t seed = 42)
{
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

// ################### DO NOT CHANGE ANYTHING ABOVE ###################

// ------------------------------------------------------------------
// Simple Kernel: C[i] = alpha*A[i] + B[i]
// ------------------------------------------------------------------
__global__ void simpleVecAdd(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N,
                             float alpha)
{
    size_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < N) {
        C[threadid] = alpha * A[threadid] + B[threadid];
    }
    // TODO: compute global thread index
    // TODO: guard against out-of-bounds
    // TODO: perform the addition
}

// ------------------------------------------------------------------
// Optimized Kernel with Grid-stride Loop 
// ------------------------------------------------------------------
__global__ void optimizedVecAdd(const float* __restrict__ A, 
                                const float* __restrict__ B, 
                                float* __restrict__ C, 
                                int N,
                                float alpha)
{
    size_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (int i = threadid; i < N; i += stride) {
        C[i] = alpha * A[i] + B[i];
    }
    // TODO: compute a global thread index
    // TODO: compute a stride
    // TODO: Implement grid-stride loop
}

// ------------------------------------------------------------------
// Optimized Kernel with Grid-stride Loop and Loop Unrolling
// ------------------------------------------------------------------
__global__ void optimizedVecAdd_lu(const float* __restrict__ A, 
                                const float* __restrict__ B, 
                                float* __restrict__ C, 
                                int N,
                                float alpha)
{
    size_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (int i = threadid; i < N; i += stride * 4) {
        if (i < N) {
            C[i] = alpha * A[i] + B[i];
        }
        if (i + stride < N) {
            C[i + stride] = alpha * A[i + stride] + B[i + stride];
        }
        if (i + 2 * stride < N) {
            C[i + 2 * stride] = alpha * A[i + 2 * stride] + B[i + 2 * stride];
        }
        if (i + 3 * stride < N) {
            C[i + 3 * stride] = alpha * A[i + 3 * stride] + B[i + 3 * stride];
        }
    }
    // TODO: compute a global thread index
    // TODO: compute a stride
    // TODO: Implement loop unrolling on top of grid-stride loop
}

// ##################################################################

int main(int argc, char** argv)
{
    // ------------------------------------------------------------------
    // Problem size
    // ------------------------------------------------------------------
    int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    if (N <= 0) {
        fprintf(stderr, "N must be positive\n");
        return EXIT_FAILURE;
    }
    float alpha = 2.5f;

    int threadsPerBlock = (argc > 2) ? std::atoi(argv[2]) : 256;
    if (threadsPerBlock <= 0) threadsPerBlock = 256;

    size_t numBytes = size_t(N) * sizeof(float);
    printf("Vector length = %d (%.2f MB per vector), TPB=%d\n",
           N, numBytes / (1024.0 * 1024.0), threadsPerBlock);
    
    // ------------------------------------------------------------------
    // Allocate & initialize host
    // ------------------------------------------------------------------
    std::vector<float> hA(N), hB(N);
    std::vector<float> hC_simple(N), hC_opt(N),
                       hC_cublas(N), hRef(N),
                       hC_opt_lu(N);

    initRandom(hA, 1234);
    initRandom(hB, 5678);

    // ------------------------------------------------------------------
    // Allocate device memory
    // ------------------------------------------------------------------
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    // TODO: cudaMalloc dA, dB, dC
    cudaMalloc((void**) &dA, numBytes);
    cudaMalloc((void**) &dB, numBytes);
    cudaMalloc((void**) &dC, numBytes);

    // Copy host -> device
    // TODO: cudaMemcpy dA, dB
    cudaMemcpy(dA, hA.data(), numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), numBytes, cudaMemcpyHostToDevice);

    // ------------------------------------------------------------------
    // Kernel config
    // ------------------------------------------------------------------
    // TODO: int numBlocks = ? // based on threadsPerBlock and N
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;


    printf("Launching %d blocks of %d threads\n", numBlocks, threadsPerBlock);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ------------------------------------------------------------------
    // simpleVecAdd
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemset(dC, 0, numBytes));
    CUDA_CHECK(cudaEventRecord(start));
    simpleVecAdd<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, N, alpha);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float timeSimpleMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&timeSimpleMs, start, stop));
    // TODO: Copy device result (hC_simple) to host
    cudaMemcpy(hC_simple.data(), dC, numBytes, cudaMemcpyDeviceToHost);

    // ------------------------------------------------------------------
    // optimizedVecAdd
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemset(dC, 0, numBytes));
    CUDA_CHECK(cudaEventRecord(start));
    // TODO: Play with the numBlocks
    optimizedVecAdd<<<4096, threadsPerBlock>>>(dA, dB, dC, N, alpha);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float timeOptimizedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&timeOptimizedMs, start, stop));
    // TODO: Copy device result (hC_opt) to host
    cudaMemcpy(hC_opt.data(), dC, numBytes, cudaMemcpyDeviceToHost);

    // ------------------------------------------------------------------
    // optimizedVecAdd_lu
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemset(dC, 0, numBytes));
    CUDA_CHECK(cudaEventRecord(start));
    // TODO: Play with the numBlocks
    optimizedVecAdd_lu<<<4096, threadsPerBlock>>>(dA, dB, dC, N, alpha);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float timeOptimizedLuMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&timeOptimizedLuMs, start, stop));
    // TODO: Copy device result (hC_opt_lu) to host
    cudaMemcpy(hC_opt_lu.data(), dC, numBytes, cudaMemcpyDeviceToHost);

    // ------------------------------------------------------------------
    // cuBLAS
    // ------------------------------------------------------------------
    double timeCublasMs = cublasVecAdd(dA, dB, hC_cublas.data(), N, alpha);
    
    // ------------------------------------------------------------------
    // Verify against CPU result
    // ------------------------------------------------------------------
    auto cpuStart = std::chrono::steady_clock::now();
    vecAddHost(hA, hB, hRef, alpha);
    auto cpuEnd = std::chrono::steady_clock::now();
    double timeCpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    printf("Simple kernel verification: ");
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(hRef[i] - hC_simple[i]);
        if (diff > 1e-5f) {
            if (errors == 0) {
                fprintf(stderr, "First mismatch %d at %d: GPU %f vs CPU %f\n",
                        errors+1, i, hC_simple[i], hRef[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf(ANSI_GREEN "Result: PASS!" ANSI_RESET "\n");
    } else {
        printf(ANSI_RED   "Result: FAIL! (mismatches: %d)" ANSI_RESET "\n", errors);
    }

    printf("Strided kernel verification: ");
    errors = 0;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(hRef[i] - hC_opt[i]);
        if (diff > 1e-5f) {
            if (errors == 0) {
                fprintf(stderr, "First mismatch %d at %d: GPU %f vs CPU %f\n",
                        errors+1, i, hC_opt[i], hRef[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf(ANSI_GREEN "Result: PASS!" ANSI_RESET "\n");
    } else {
        printf(ANSI_RED   "Result: FAIL! (mismatches: %d)" ANSI_RESET "\n", errors);
    }

    printf("Strided kernel with loop unrolling verification: ");
    errors = 0;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(hRef[i] - hC_opt_lu[i]);
        if (diff > 1e-5f) {
            if (errors == 0) {
                fprintf(stderr, "First mismatch %d at %d: GPU %f vs CPU %f\n",
                        errors+1, i, hC_opt_lu[i], hRef[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf(ANSI_GREEN "Result: PASS!" ANSI_RESET "\n");
    } else {
        printf(ANSI_RED   "Result: FAIL! (mismatches: %d)" ANSI_RESET "\n", errors);
    }

    // ------------------------------------------------------------------
    // Timing summary
    // ------------------------------------------------------------------
    // Fixed-width header
    printf("%-10s %-8s %-15s %-15s %-20s %-12s %-12s\n",
        "N", "TPB", "Simple_ms", "Strided_ms", "StridedUnroll_ms", "cuBLAS_ms", "CPU_ms");

    // Fixed-width row
    printf("%-10d %-8d %-15.3f %-15.3f %-20.3f %-12.3f %-12.3f\n",
        N, threadsPerBlock,
        timeSimpleMs,
        timeOptimizedMs,
        timeOptimizedLuMs,
        timeCublasMs,
        timeCpuMs);

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    // TODO: Free device memory
    // free dA, dB, dC
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
