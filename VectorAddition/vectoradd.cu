#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

// Macro to check for CUDA errors
#define CHECK_CUDA(call)                                             \
do {                                                                 \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr,                                              \
                "CUDA Error %d at %s:%d\n",                          \
                (int)err, __FILE__, __LINE__);                       \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while (0)

// Initialize a vector with random float values between 0 and 1
void initRandom(std::vector<float>& v) {
    for (auto& x : v)
        x = static_cast<float>(rand()) / RAND_MAX;
}

// CPU reference implementation of vector addition
void vecAddHost(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, float alpha) {
    for (size_t i = 0; i < A.size(); i++)
        C[i] = alpha * A[i] + B[i];
}

// Verify the result of GPU computation against CPU reference
bool verify(const std::vector<float>& ref, const std::vector<float>& result) {
    for (size_t i = 0; i < ref.size(); i++) {
        if (std::abs(ref[i] - result[i]) > 1e-5f)
            return false;
    }
    return true;
}

// Simple GPU vector addition kernel
__global__ void simpleVecAdd(const float* A, const float* B, float* C, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = alpha * A[idx] + B[idx];
}

// Optimized GPU vector addition kernel with grid-stride loop
__global__ void optimizedVecAdd(const float* A, const float* B, float* C, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N; i += stride)
        C[i] = alpha * A[i] + B[i];
}

// Optimized GPU vector addition kernel with loop unrolling
__global__ void optimizedVecAdd_lu(const float* A, const float* B, float* C, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N; i += stride * 4) { // Unroll the loop by a factor of 4 chosen arbitrarily
        if (i < N)
            C[i] = alpha * A[i] + B[i];
        if (i + stride < N)
            C[i + stride] = alpha * A[i + stride] + B[i + stride];
        if (i + 2 * stride < N)
            C[i + 2 * stride] = alpha * A[i + 2 * stride] + B[i + 2 * stride];
        if (i + 3 * stride < N)
            C[i + 3 * stride] = alpha * A[i + 3 * stride] + B[i + 3 * stride];
    }
}

int main(int argc, char** argv)
{
    int deviceCount; // Get the number of CUDA devices available
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount)); // Check for errors in getting the device count
    printf("CUDA devices found: %d\n", deviceCount); // Print the number of CUDA devices found
    
    // Set the device to use, defaulting to device 0 if not specified
    int device = (argc > 3) ? std::atoi(argv[3]) : 0;
    CHECK_CUDA(cudaSetDevice(device));

    int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);

    // Set the number of threads per block, defaulting to 256 if not specified
    int threadsPerBlock = (argc > 2) ? std::atoi(argv[2]) : 256;

    // Set the scalar multiplier for the vector addition
    float alpha = 2.5f;

    // Calculate the number of bytes needed for the vectors
    size_t numBytes = N * sizeof(float);

    // Allocate host vectors
    std::vector<float> hA(N), hB(N);
    std::vector<float> hC_simple(N), hC_opt(N), hC_opt_lu(N), hRef(N);

    // Initialize host vectors with random values
    initRandom(hA);
    initRandom(hB);

    // Allocate device vectors and allocate memory on the GPU
    float *dA, *dB, *dC;
    cudaMalloc(&dA, numBytes);
    cudaMalloc(&dB, numBytes);
    cudaMalloc(&dC, numBytes);

    // Copy host vectors to device memory
    cudaMemcpy(dA, hA.data(), numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), numBytes, cudaMemcpyHostToDevice);

    // Calculate the number of blocks needed
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Perform simple vector addition on the GPU
    cudaMemset(dC, 0, numBytes);
    cudaEventRecord(start);
    simpleVecAdd<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, N, alpha);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    float timeSimpleMs = 0;
    cudaEventElapsedTime(&timeSimpleMs, start, stop);

    // Copy the result from device to host
    cudaMemcpy(hC_simple.data(), dC, numBytes, cudaMemcpyDeviceToHost);

    // Perform optimized vector addition on the GPU
    cudaMemset(dC, 0, numBytes);
    cudaEventRecord(start);
    optimizedVecAdd<<<4096, threadsPerBlock>>>(dA, dB, dC, N, alpha);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeOptimizedMs = 0;
    cudaEventElapsedTime(&timeOptimizedMs, start, stop);

    // Copy the result from device to host
    cudaMemcpy(hC_opt.data(), dC, numBytes, cudaMemcpyDeviceToHost);

    // Perform optimized vector addition with loop unrolling on the GPU
    cudaMemset(dC, 0, numBytes);
    cudaEventRecord(start);
    optimizedVecAdd_lu<<<4096, threadsPerBlock>>>(dA, dB, dC, N, alpha);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeOptimizedLuMs = 0;
    cudaEventElapsedTime(&timeOptimizedLuMs, start, stop);

    // Copy the result from device to host
    cudaMemcpy(hC_opt_lu.data(), dC, numBytes, cudaMemcpyDeviceToHost);

    // Perform vector addition on the CPU for reference
    auto cpuStart = std::chrono::steady_clock::now();
    vecAddHost(hA, hB, hRef, alpha);
    auto cpuEnd = std::chrono::steady_clock::now();
    double timeCpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    // Output the verification results and timings for each kernel
    printf("Simple Kernel       : %s\n",
           verify(hRef, hC_simple) ? "PASS" : "FAIL");

    printf("Grid-Stride Kernel  : %s\n",
           verify(hRef, hC_opt) ? "PASS" : "FAIL");

    printf("Loop-Unrolled Kernel: %s\n",
           verify(hRef, hC_opt_lu) ? "PASS" : "FAIL");

    printf("\nPerformance Results\n");
    printf("----------------------------------\n");

    printf("Simple Kernel       : %.3f ms\n", timeSimpleMs);
    printf("Grid-Stride Kernel  : %.3f ms\n", timeOptimizedMs);
    printf("Loop-Unrolled Kernel: %.3f ms\n", timeOptimizedLuMs);
    printf("CPU Reference       : %.3f ms\n", timeCpuMs);

    // Free device memory and destroy CUDA events
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}