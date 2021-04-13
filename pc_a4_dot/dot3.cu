#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 8 //33 * 1024 * 1024
__global__ void dot(int *a, int *b, int *c, int *dot)
{
    int i;
    int tid = threadIdx.x;
    c[tid] = a[tid] * b[tid];
    // __syncthreads();
    while (tid < N) {
        i = blockDim.x / 2;
        while (i != 0)
        {
            if ((tid / blockDim.x) < i)
            {
                // c[tid] += c[tid + i];
                printf("%d %d %d\n", i, tid, tid+i);
            }
            __syncthreads();
            i = i / 2;
        }
        printf("%d\n", tid);
        // if (tid == flag)
        // {
        //     *dot = c[tid];
        // }
        tid += blockDim.x;
    }
}

int main(void)
{
    int *a, *b;
    int *dev_a, *dev_b, *dev_c, *dev_dot;
    int dotCPU = 0;
    int dotGPU;

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    a = (int *)malloc(N * sizeof(int));
    b = (int *)malloc(N * sizeof(int));

    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * sizeof(int));
    cudaMalloc((void **)&dev_dot, sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % 256;
        b[i] = rand() % 256;
    }
    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    dot<<<1, 4>>>(dev_a, dev_b, dev_c, dev_dot);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);

    // copy the array 'dev_dot' back from the GPU to the CPU
    cudaMemcpy(&dotGPU, dev_dot, sizeof(int), cudaMemcpyDeviceToHost);

    // verify that the GPU did the work we requested
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

    // free the memory allocated on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_dot);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
