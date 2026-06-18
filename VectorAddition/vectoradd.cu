#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

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
void initRandom(std::vector<float>& v) {
    for (auto& x : v)
        x = static_cast<float>(rand()) / RAND_MAX;
}

void vecAddHost(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                float alpha) {
    for (size_t i = 0; i < A.size(); i++)
        C[i] = alpha * A[i] + B[i];
}

bool verify(const std::vector<float>& ref,
            const std::vector<float>& result) {
    for (size_t i = 0; i < ref.size(); i++) {
        if (std::abs(ref[i] - result[i]) > 1e-5f)
            return false;
    }
    return true;
}

__global__ void simpleVecAdd(const float* A, const float* B,
                             float* C, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = alpha * A[idx] + B[idx];
}

__global__ void optimizedVecAdd(const float* A, const float* B,
                                float* C, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N; i += stride)
        C[i] = alpha * A[i] + B[i];
}

__global__ void optimizedVecAdd_lu(const float* A, const float* B,
                                   float* C, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N; i += stride * 4) {
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
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    printf("CUDA devices found: %d\n", deviceCount);

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    
    int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    int threadsPerBlock = (argc > 2) ? std::atoi(argv[2]) : 256;

    float alpha = 2.5f;
    size_t numBytes = N * sizeof(float);

    std::vector<float> hA(N), hB(N);
    std::vector<float> hC_simple(N), hC_opt(N), hC_opt_lu(N), hRef(N);

    initRandom(hA);
    initRandom(hB);

    float *dA, *dB, *dC;

    cudaMalloc(&dA, numBytes);
    cudaMalloc(&dB, numBytes);
    cudaMalloc(&dC, numBytes);
    // printf("malloc dA: %s\n", cudaGetErrorString(err));

    cudaMemcpy(dA, hA.data(), numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), numBytes, cudaMemcpyHostToDevice);

    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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

    cudaMemcpy(hC_simple.data(), dC, numBytes, cudaMemcpyDeviceToHost);

    cudaMemset(dC, 0, numBytes);
    cudaEventRecord(start);

    optimizedVecAdd<<<4096, threadsPerBlock>>>(dA, dB, dC, N, alpha);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeOptimizedMs = 0;
    cudaEventElapsedTime(&timeOptimizedMs, start, stop);

    cudaMemcpy(hC_opt.data(), dC, numBytes, cudaMemcpyDeviceToHost);

    cudaMemset(dC, 0, numBytes);
    cudaEventRecord(start);

    optimizedVecAdd_lu<<<4096, threadsPerBlock>>>(dA, dB, dC, N, alpha);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeOptimizedLuMs = 0;
    cudaEventElapsedTime(&timeOptimizedLuMs, start, stop);

    cudaMemcpy(hC_opt_lu.data(), dC, numBytes, cudaMemcpyDeviceToHost);

    auto cpuStart = std::chrono::steady_clock::now();
    vecAddHost(hA, hB, hRef, alpha);
    auto cpuEnd = std::chrono::steady_clock::now();
    printf("\nSample values:\n");
    for (int i = 0; i < 5; i++) {
        printf("i=%d CPU=%f GPU=%f\n",
            i,
            hRef[i],
            hC_simple[i]);
    }

    double timeCpuMs =
        std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

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

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}