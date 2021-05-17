#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#define WIDTH 1024
#define TILE_WIDTH 16
#define BLOCKSPERGRID WIDTH / TILE_WIDTH

int N[WIDTH][WIDTH] = {0};
int T[WIDTH][WIDTH] = {0};

__global__ void transpose(int *Nd, int *Td);
__device__ int GetElement(int *matrix, int row, int col);
__device__ void SetElement(int *matrix, int row, int col, int value);
__device__ int *GetSubMatrix(int *matrix, int blockrow, int blockcol);

int main(int argc, char *argv[])
{
    float elapsedTime;
    
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            N[i][j] = (int) (rand() % 255 + 1);
        }
    }

    // Original
    size_t size = WIDTH * WIDTH * sizeof(int);
    int *Nd, *Td;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&Nd, size);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&Td, size);

    struct timeval starttime, endtime;
    gettimeofday(&starttime, NULL);
    for (int i = 0; i < WIDTH; i++)
    {
        for (int j = i + 1; j < WIDTH; j++)
        {
            int temp = N[i][j];
            N[i][j] = N[j][i];
            N[j][i] = temp;
        }
    }
    gettimeofday(&endtime, NULL);
    double executime;
    executime = (endtime.tv_sec - starttime.tv_sec) * 1000.0;
    executime += (endtime.tv_usec - starttime.tv_usec) / 1000.0;
    printf("CPU time: %13lf msec\n", executime);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(BLOCKSPERGRID, BLOCKSPERGRID);

    cudaEventRecord(start, 0);
    transpose<<<dimGrid, dimBlock>>>(Nd, Td);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);

    cudaError_t cuda_err = cudaGetLastError();
    if (cudaSuccess != cuda_err) {
        printf("before kernel call: error = %s\n", cudaGetErrorString (cuda_err));
        exit(1);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(T, Td, size, cudaMemcpyDeviceToHost);

    int pass = 1;
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            if (N[i][j] != T[i][j]) {
                printf("N[%d][%d] = %d   T[%d][%d] = %d\n", i, j, N[i][j], i, j, T[i][j]);
                pass = 0;
                break;
            }
        }
    }
    printf("Test %s\n", (pass)?"PASSED":"FAILED");
    cudaFree(Nd);
    cudaFree(Td);

    return 0;
}

__global__ void transpose(int *Nd, int *Td) {
    int xIndex = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int yIndex = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int index_in = xIndex + WIDTH * yIndex;
    int index_out = yIndex + WIDTH * xIndex;
    Td[index_out] = Nd[index_in];
}

__device__ int GetElement(int *matrix, int y, int x)
{
    return *(matrix + y * WIDTH + x);
}

__device__ void SetElement(int *matrix, int y, int x, int value)
{
    *(matrix + y * WIDTH + x) = value;
}

__device__ int *GetSubMatrix(int *matrix, int block_y, int block_x)
{
    return (matrix + block_y * TILE_WIDTH * WIDTH + block_x * TILE_WIDTH);
}
