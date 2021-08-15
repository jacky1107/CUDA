#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 12
#define threadsPerBlock 4

__global__ void dot(int *a, int *b, int *c, int *dot)
{
    int tid = threadIdx.x;
    int temp_tid = tid;
    int i;
    c[tid] = 0;
    while (temp_tid < N)
    {
        printf("%d %d\n", tid, temp_tid);
        c[tid] += a[temp_tid] * b[temp_tid];
        temp_tid += blockDim.x;
    }
    __syncthreads();
    i = blockDim.x / 2;
    while (i != 0)
    {
        if (tid < i)
        {
            c[tid] += c[tid + i]; //reduction add
        }
        __syncthreads();
        i /= 2;
    }
    if (tid == 0)
        *dot = c[tid];
}

int main(void)
{
    int *a, *b;
    int *dev_a, *dev_b, *dev_c, *dev_dot;
    int dotCPU = 0;
    int dotGPU;
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

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    dot<<<1, threadsPerBlock>>>(dev_a, dev_b, dev_c, dev_dot);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %f msec\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(&dotGPU, dev_dot, sizeof(int), cudaMemcpyDeviceToHost);
    bool success = true;
    struct timespec t_start, t_end;
    double elapsedTimeCPU;
    clock_gettime(CLOCK_REALTIME, &t_start);
    for (int i = 0; i < N; i++)
    {
        dotCPU += a[i] * b[i];
    }
    clock_gettime(CLOCK_REALTIME, &t_end);
    elapsedTimeCPU = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    elapsedTimeCPU += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    printf("CPU elapsedTime: %lf ms\n", elapsedTimeCPU);
    printf("Speedup: %.2lf\n", elapsedTimeCPU / elapsedTime);
    if (dotCPU != dotGPU)
    {
        printf("Error: dotCPU %d != dotGPU %d\n", dotCPU, dotGPU);
        success = false;
    }
    if (success)
        printf("Test pass!\n");
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_dot);
    return 0;
}