#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_ROWS 10000
#define NUM_COLS 1000

int a[NUM_ROWS][NUM_COLS];
int b[NUM_ROWS][NUM_COLS];
int c[NUM_ROWS][NUM_COLS];

__global__ void add(int *a, int *b, int *c)
{
    int x = blockIdx.x;
    int y = gridDim.x * blockIdx.y;
    int tid = x + y;
    while (tid < NUM_ROWS * NUM_COLS) {
        c[tid] = a[tid] + b[tid];
        tid += gridDim.x * gridDim.y;
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
    for(int i=0;i<NUM_ROWS;i++)
    {
        for(int j=0;j<NUM_COLS;j++)
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

    int R = NUM_ROWS / 32;
    int C = NUM_COLS / 32;
    dim3 dimGrid (R, C, 1);
    dim3 dimBlock (1, 1, 1);

    cudaEventRecord(start, 0);
    add<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);

    cuError = cudaGetLastError();
    if (cudaSuccess != cuError)
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