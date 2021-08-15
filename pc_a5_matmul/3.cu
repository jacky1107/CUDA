#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#define WIDTH 1024
#define TILE_WIDTH 16
#define BLOCKSPERGRID (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH

int M[WIDTH][WIDTH] = {0};
int N[WIDTH][WIDTH] = {0};
int P[WIDTH][WIDTH] = {0};
int MxN[WIDTH][WIDTH] = {0};

__global__ void mat_mul(int *Md, int *Nd, int *Pd);
__device__ int GetElement(int *matrix, int row, int col);
__device__ void SetElement(int *matrix, int row, int col, int value);
__device__ int *GetSubMatrix(int *matrix, int blockrow, int blockcol);

int main(int argc, char *argv[])
{
    float elapsedTime;

    for (int i = 0; i < WIDTH; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            M[i][j] = (int)(rand() % 255 + 1);
            N[i][j] = (int)(rand() % 255 + 1);
        }
    }

    struct timeval starttime, endtime;
    gettimeofday(&starttime, NULL);
    for (int i = 0; i < WIDTH; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            for (int k = 0; k < WIDTH; ++k)
            {
                MxN[i][j] += M[i][k] * N[k][j];
            }
        }
    }
    gettimeofday(&endtime, NULL);
    double executime;
    executime = (endtime.tv_sec - starttime.tv_sec) * 1000.0;
    executime += (endtime.tv_usec - starttime.tv_usec) / 1000.0;
    printf("CPU time: %13lf msec\n", executime);

    // Original
    size_t size = WIDTH * WIDTH * sizeof(int);
    int *Md, *Nd, *Pd, *Td;
    cudaMalloc((void **)&Md, size);
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&Nd, size);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&Td, size);
    cudaMalloc((void **)&Pd, size);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(BLOCKSPERGRID, BLOCKSPERGRID);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    mat_mul<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);

    cudaError_t cuda_err = cudaGetLastError();
    if (cudaSuccess != cuda_err)
    {
        printf("before kernel call: error = %s\n", cudaGetErrorString(cuda_err));
        exit(1);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
    int pass = 1;
    for (int i = 0; i < WIDTH; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            if (MxN[i][j] != P[i][j])
            {
                printf("MxN[%d][%d] = %d   P[%d][%d] = %d\n", i, j, MxN[i][j], i, j, P[i][j]);
                pass = 0;
                break;
            }
        }
    }
    printf("Test %s\n", (pass) ? "PASSED" : "FAILED");
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);

    return 0;
}

__global__ void mat_mul(int *Md, int *Nd, int *Pd)
{
    int block_y = blockIdx.y;
    int block_x = blockIdx.x;

    int *Pd_sub = GetSubMatrix(Pd, block_y, block_x);

    int y = threadIdx.y;
    int x = threadIdx.x;

    int Pvalue = 0;

    for (int m = 0; m < (WIDTH / TILE_WIDTH); ++m)
    {
        int *Md_sub = GetSubMatrix(Md, block_y, m);
        int *Nd_sub = GetSubMatrix(Nd, m, block_x);

        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            int Melement = GetElement(Md_sub, y, k);
            int Nelement = GetElement(Nd_sub, k, x);
            Pvalue += Melement * Nelement;
        }

        __syncthreads();
    }

    SetElement(Pd_sub, y, x, Pvalue);
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
