#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 33 * 1024 * 1024
#define THREADSPERBLOCK 32
#define BLOCKSPERGRID (N + THREADSPERBLOCK - 1) / THREADSPERBLOCK

__global__ void dot(int *a, int *b, int *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    __shared__ int cache[THREADSPERBLOCK];
    int cacheIndex = threadIdx.x;

    cache[cacheIndex] = a[tid] * b[tid];
    __syncthreads();

    i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

int main(void)
{
    int *a, *b, *partial_c;
    int *dev_a, *dev_b, *partial_dev_c;
    int dotCPU = 0;
    int dotGPU = 0;

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    a = (int *)malloc(N * sizeof(int));
    b = (int *)malloc(N * sizeof(int));
    partial_c = (int *)malloc(BLOCKSPERGRID * sizeof(int));

    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&partial_dev_c, BLOCKSPERGRID * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % 256;
        b[i] = rand() % 256;
    }
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    dot<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(dev_a, dev_b, partial_dev_c);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);

    cudaMemcpy(partial_c, partial_dev_c, BLOCKSPERGRID * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < BLOCKSPERGRID; i++)
    {
        dotGPU += partial_c[i];
    }

    bool success = true;
    for (int i = 0; i < N; i++)
    {
        dotCPU += a[i] * b[i];
    }
    if (dotCPU != dotGPU)
    {
        printf("Error: dotCPU %d != dotGPU %d\n", dotCPU, dotGPU);
        success = false;
    }
    if (success)
        printf("Test pass!\n");
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(partial_dev_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}