#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_ROWS 10000
#define NUM_COLS 1000

int a[NUM_ROWS][NUM_COLS];
int b[NUM_ROWS][NUM_COLS];
int c[NUM_ROWS][NUM_COLS];

__global__ void add(int *a, int *b, int *c)
{
    int tx = threadIdx.x;
    int ty = blockDim.x * threadIdx.y;
    int bx = blockDim.x * blockDim.y * blockIdx.x;
    int by = gridDim.x * (blockDim.x * blockDim.y) * blockIdx.y;
    int tid = bx + by + tx + ty;
    while (tid < NUM_ROWS * NUM_COLS) {
        c[tid] = a[tid] + b[tid];
        tid = tid + gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    }
}

int main()
{
    int *dev_a, *dev_b, *dev_c;
    int size = NUM_ROWS * NUM_COLS * sizeof(int);
    cudaError_t cuError = cudaSuccess ;

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int i = 0; i < NUM_ROWS; i++)
    {
        for(int j = 0; j < NUM_COLS; j++)
        {
            a[i][j] = rand() % size + 1;
            b[i][j] = rand() % size + 1;
        }
    }
    if (cudaMalloc((void **)&dev_a, size) != cudaSuccess) return 1;
    if (cudaMalloc((void **)&dev_b, size) != cudaSuccess) return 1;
    if (cudaMalloc((void **)&dev_c, size) != cudaSuccess) return 1;
    if (cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    
    int per_threads_x = 32;
    int per_threads_y = 32;
    int per_blocks_x = NUM_ROWS / 32;
    int per_blocks_y = NUM_COLS / 32;
    printf("%d %d\n", per_blocks_x, per_blocks_y);

    dim3 dimBlock (per_threads_x, per_threads_y, 1);
    dim3 dimGrid (per_blocks_x, per_blocks_y, 1);

    cudaEventRecord(start, 0);
    add<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);

    if (cudaGetLastError() != cudaSuccess)
    {
        printf ("Failed in kernel launch and reason is %s\n", cudaGetErrorString(cuError)) ;
        return 1 ;
    }
    if (cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost) != cudaSuccess) return 1 ;

    bool success = true;
    for (int i = 0; i < NUM_ROWS; i++)
    {
        for (int j = 0; j < NUM_COLS; j++)
        {
            if ((a[i][j] + b[i][j]) != c[i][j])
            {
                success = false;
                break;
            }

        }
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