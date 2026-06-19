#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(stmt) do { cudaError_t err = (stmt); if (err != cudaSuccess) { fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(err)); std::exit(EXIT_FAILURE); } } while (0)

#ifndef NUM_BINS
#define NUM_BINS 256
#endif

#define COARSE_FACTOR 64

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    __shared__ int Bgv_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        Bgv_s[bin] = 0;
    }
    __syncthreads();

    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx < width * height) {
        atomicAdd(&Bgv_s[image[tx]], 1);
    }
    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = Bgv_s[bin];
        if (binValue > 0) {
            atomicAdd(&bins[bin], binValue);
        }
    } 
}

void histogram_gpu_private(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    unsigned int numThreadsPerBlock = 1024;
    unsigned int total = width * height;
    unsigned int numBlocks = (total + numThreadsPerBlock - 1) / numThreadsPerBlock;
    histogram_private_kernel<<<numBlocks, numThreadsPerBlock>>>(image, bins, width, height);
}

__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    __shared__ unsigned int Bgv_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += COARSE_FACTOR) {
        Bgv_s[bin] = 0;
    }
    __syncthreads();

    int segmentid = blockIdx.x * (COARSE_FACTOR * blockDim.x);
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int idx = segmentid + threadIdx.x + (c * blockDim.x);
        if (idx < width * height) {
            atomicAdd(&Bgv_s[image[idx]], 1u);
        }
    }
    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = Bgv_s[bin];
        if (binValue > 0) {
            atomicAdd(&bins[bin], binValue);
        }
    }  
}

void histogram_gpu_private_coarse(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    unsigned int numThreadsPerBlock = 256;
    unsigned int total = width * height;
    unsigned int numBlocks = ((total + (COARSE_FACTOR * numThreadsPerBlock) - 1) / (COARSE_FACTOR * numThreadsPerBlock));
    histogram_private_coarse_kernel<<<numBlocks, numThreadsPerBlock>>>(image, bins, width, height);
}

static void cpu_histogram(const unsigned char* img, unsigned int* bins, unsigned int n) {
    for (unsigned int b = 0; b < NUM_BINS; ++b) bins[b] = 0u;
    for (unsigned int i = 0; i < n; ++i) bins[img[i]]++;
}

static void init_random_u8(unsigned char* v, unsigned int n, unsigned long long seed) {
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (unsigned int i = 0; i < n; ++i) v[i] = (unsigned char)dist(gen);
}

static int verify_equal(const unsigned int* a, const unsigned int* b, int n){
    for (int i = 0; i < n; ++i){
        if (a[i] != b[i]) {
            fprintf(stderr, "Mismatch bin %d: %u vs %u\n", i, a[i], b[i]);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char** argv){
    if (argc < 3){
        fprintf(stderr, "Usage: %s <width> <height> [repeat]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int w_in = std::atoi(argv[1]);
    int h_in = std::atoi(argv[2]);
    int repeat = (argc >= 4) ? std::atoi(argv[3]) : 10;
    if (w_in <= 0 || h_in <= 0) return EXIT_FAILURE;
    if (repeat < 1) repeat = 1;

    unsigned int width = w_in;
    unsigned int height = h_in;
    unsigned int N = width * height;

    std::vector<unsigned char> h_img(N);
    std::vector<unsigned int> h_ref(NUM_BINS), h_gpu_private(NUM_BINS), h_gpu_coarse(NUM_BINS);

    init_random_u8(h_img.data(), N, 1234ULL);

    auto t0 = std::chrono::steady_clock::now();
    cpu_histogram(h_img.data(), h_ref.data(), N);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    unsigned char *d_img = 0;
    unsigned int *d_bins = 0;
    CUDA_CHECK(cudaMalloc((void**) &d_img, N));
    CUDA_CHECK(cudaMalloc((void**) &d_bins, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(), N * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));
    histogram_gpu_private(d_img, d_bins, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float ms_private = 0.0f;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));
        CUDA_CHECK(cudaEventRecord(start));
        histogram_gpu_private(d_img, d_bins, width, height);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
        ms_private += t;
    }
    ms_private /= repeat;

    CUDA_CHECK(cudaMemcpy(h_gpu_private.data(), d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    int e1 = verify_equal(h_ref.data(), h_gpu_private.data(), NUM_BINS);

    CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));
    histogram_gpu_private_coarse(d_img, d_bins, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms_coarse = 0.0f;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));
        CUDA_CHECK(cudaEventRecord(start));
        histogram_gpu_private_coarse(d_img, d_bins, width, height);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));
        ms_coarse += t;
    }
    ms_coarse /= repeat;

    CUDA_CHECK(cudaMemcpy(h_gpu_coarse.data(), d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    int e2 = verify_equal(h_ref.data(), h_gpu_coarse.data(), NUM_BINS);

    unsigned long long s_ref = 0, s1 = 0, s2 = 0;
    for (int i = 0; i < NUM_BINS; ++i) { 
        s_ref += h_ref[i]; 
        s1 += h_gpu_private[i]; 
        s2 += h_gpu_coarse[i]; 
    }

    printf("Image: %ux%u | Pixels=%u | Bins=%d | Repeat=%d\n", width, height, N, NUM_BINS, repeat);
    printf("CPU: %.3f ms | CPU sum=%llu\n", cpu_ms, s_ref);
    printf("[private] %s | avg %.3f ms | sum=%llu\n", (e1 == 0 ? "PASS" : "FAIL"), ms_private, s1);
    printf("[coarse ] %s | avg %.3f ms | sum=%llu\n", (e2 == 0 ? "PASS" : "FAIL"), ms_coarse, s2);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_bins));

    return (e1 | e2) ? EXIT_FAILURE : EXIT_SUCCESS;
}