#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 10000000

__global__ void add(int *a, int *b, int *c)
{
    int tid = blockIdx.x;
    while (tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += gridDim.x;
    }
}

int main(void)
{
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    a = (int *)malloc(N * sizeof(int));
    b = (int *)malloc(N * sizeof(int));
    c = (int *)malloc(N * sizeof(int));

    if (cudaMalloc((void **)&dev_a, N * sizeof(int)) != cudaSuccess) return 1;
    if (cudaMalloc((void **)&dev_b, N * sizeof(int)) != cudaSuccess) return 1 ;
    if (cudaMalloc((void **)&dev_c, N * sizeof(int)) != cudaSuccess) return 1 ;
    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % N;
        b[i] = rand() % N;
    }
    
    int *d;
    d = (int *)malloc(N * sizeof(int));
    int cpu = true;
    double elapsedTimeCPU;
    struct timespec t_start, t_end;
    if (cpu) {
        clock_gettime( CLOCK_REALTIME, &t_start);
        for (int i = 0; i < N; i++) {
            d[i] = a[i] + b[i];
        }
        clock_gettime( CLOCK_REALTIME, &t_end);
        elapsedTimeCPU = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
        elapsedTimeCPU += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
        printf("CPU elapsedTime: %lf ms\n", elapsedTimeCPU);
    }

    cudaEventRecord(start, 0);
    if (cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return 1 ;
    if (cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return 1 ;

    int per_blocks = 32768;// N / (256 * 8);
    printf("Per blocks: %d\n", per_blocks);
    
    add<<<per_blocks, 1>>>(dev_a, dev_b, dev_c);
    cudaEventRecord(stop, 0);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);

    if (cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 1 ;
    
    bool success = true;
    for (int i = 0; i < N; i++)
        if ((a[i] + b[i]) != c[i])
        {
            success = false;
            break;
        }
    if (success)
        printf("We did it!\n");
    else
        printf("Failed\n");

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}