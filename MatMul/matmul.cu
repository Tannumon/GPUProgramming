#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

#define CUDA_CHECK(stmt) do { \
    cudaError_t err = (stmt); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR %s (%d): %s\n", \
                #stmt, int(err), cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static void cpu_sgemm(const float *A, const float *B, float *C,
                      int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

static void init_random(float* v, long long n, unsigned long long seed = 42ULL) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (long long i = 0; i < n; i++) v[i] = dist(gen);
}

__global__ void simple_gemm(const float* A, const float* B, float* C,
                            int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

#define TILE 32 // Adjust tile size based on your GPU's shared memory capacity usually 32
__global__ void tiled_gemm_sm(const float* A, const float* B, float* C,
                              int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {

        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

int main()
{
    const int TESTNUM = 6;
    const int repeat = 10;

    // Diverse matrix sizes, including non-multiples of TILE to test edge cases
    const unsigned int M_list[] = {512,129,191,255,383,511};
    const unsigned int N_list[] = {512,131,193,257,385,513};
    const unsigned int K_list[] = {512,1001,1023,1025,1151,1277};

    double times[TESTNUM][3]; // CPU, simple, tiled
    std::vector<std::string> names;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int pass = 0;

    for (int i = 0; i < TESTNUM; i++) {

        int M = M_list[i], N = N_list[i], K = K_list[i];
        names.push_back(std::to_string(M) + "-" + std::to_string(K) + "-" + std::to_string(N));

        size_t sA = M * K * sizeof(float);
        size_t sB = K * N * sizeof(float);
        size_t sC = M * N * sizeof(float);

        float *hA = new float[M*K];
        float *hB = new float[K*N];
        float *hC_ref = new float[M*N];
        float *hC_s = new float[M*N];
        float *hC_t = new float[M*N];

        init_random(hA, M*K);
        init_random(hB, K*N);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, sA));
        CUDA_CHECK(cudaMalloc(&dB, sB));
        CUDA_CHECK(cudaMalloc(&dC, sC));

        CUDA_CHECK(cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice));

        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_sgemm(hA, hB, hC_ref, M, N, K);
        auto t2 = std::chrono::high_resolution_clock::now();

        double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

        CUDA_CHECK(cudaMemset(dC, 0, sC));
        CUDA_CHECK(cudaEventRecord(start));

        for (int r = 0; r < repeat; r++)
            simple_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float simple_ms;
        CUDA_CHECK(cudaEventElapsedTime(&simple_ms, start, stop));
        simple_ms /= repeat;

        CUDA_CHECK(cudaMemcpy(hC_s, dC, sC, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemset(dC, 0, sC));
        CUDA_CHECK(cudaEventRecord(start));

        for (int r = 0; r < repeat; r++)
            tiled_gemm_sm<<<grid, block>>>(dA, dB, dC, M, N, K);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float tiled_ms;
        CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, start, stop));
        tiled_ms /= repeat;

        CUDA_CHECK(cudaMemcpy(hC_t, dC, sC, cudaMemcpyDeviceToHost));

        auto check = [&](float* a, float* b) {
            for (int i = 0; i < M*N; i++)
                if (fabs(a[i] - b[i]) > 1e-2f) return 1;
            return 0;
        };

        int ok = !check(hC_ref, hC_s) && !check(hC_ref, hC_t);
        pass += ok;

        printf("[%s]: simple=%s tiled=%s\n",
               names[i].c_str(),
               check(hC_ref, hC_s) ? "FAIL" : "PASS",
               check(hC_ref, hC_t) ? "FAIL" : "PASS");

        times[i][0] = cpu_ms;
        times[i][1] = simple_ms;
        times[i][2] = tiled_ms;

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        delete[] hA; delete[] hB;
        delete[] hC_ref; delete[] hC_s; delete[] hC_t;
    }

    printf("\n[%d/%d] %s\n", pass, TESTNUM,
           pass == TESTNUM ? "PASS" : "FAIL");

    printf("M-K-N | CPU | Simple | Tiled\n");
    for (int i = 0; i < TESTNUM; i++) {
        printf("%s | %.3f | %.3f | %.3f\n",
               names[i].c_str(),
               times[i][0], times[i][1], times[i][2]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}