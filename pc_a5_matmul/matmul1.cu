#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#define WIDTH 256

__global__ void MatMulKernel(float *Md, float *Nd, float *Pd, int width);

int main(int argc, char *argv[])
{
    int width = WIDTH;
    float M[WIDTH][WIDTH] = {0};
    float N[WIDTH][WIDTH] = {0};
    float P[WIDTH][WIDTH] = {0};
    float MxN[WIDTH][WIDTH] = {0};
    int pass = 1;
    
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            M[i][j] = rand() % 30;
            N[i][j] = rand() % 30;
        }
    }
    
    struct timeval starttime, endtime;
    gettimeofday(&starttime, NULL);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < width; ++k) {
                MxN[i][j] += M[i][k] * N[k][j];
            }
        }
    }
    gettimeofday(&endtime, NULL);
    double executime;
    executime = (endtime.tv_sec - starttime.tv_sec) * 1000.0;
    executime += (endtime.tv_usec - starttime.tv_usec) / 1000.0;
    printf("CPU time: %13lf msec\n", executime);
    
	size_t size = width * width * sizeof(float);
    float *Md, *Nd, *Pd;
    
    // Allocate and Load M, N to device memory
    cudaMalloc((void **)&Md, size);
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    
    cudaMalloc((void **)&Nd, size);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
    
    // Allocate P on the device
    cudaMalloc((void **)&Pd, size);
    
    // Setup the execution configuration
    dim3 dimGrid(1, 1);
    dim3 dimBlock(32, 32);
    
    // Get start time event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // Invoke kernel
    MatMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);
    cudaError_t cuda_err = cudaGetLastError();
    if ( cudaSuccess != cuda_err ){
        printf("before kernel call: error = %s\n", cudaGetErrorString (cuda_err));
        exit(1) ;
    }
    
    // Get stop time event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // Compute execution time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Read P from device memory
    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
    
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            if((MxN[i][j] - P[i][j]) > 0.0001) {
				// printf("MxN[%d][%d] = %2.0f   P[%d][%d] = %2.0f\n", i, j, MxN[i][j], i, j, P[i][j]);
                pass = 0;
                break;
            }
        }
    }
    
    printf("Test %s\n", (pass)?"PASSED":"FAILED");
    
    return 0;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(float *Md, float *Nd, float *Pd, int width)
{
    int i = threadIdx.y;
    int j = threadIdx.x;
    int tid_i = i;
    int tid_j = j;
    while(tid_i < width) {
        while(tid_j < width) {
            float Pvalue = 0;
            for (int k = 0; k < width; ++k) {
                int x = tid_i * width + k;
                int y = k * width + tid_j;
                float Melement = *(Md + x);
                float Nelement = *(Nd + y);
                Pvalue += Melement * Nelement;
            }
            *(Pd + tid_i * width + tid_j) = Pvalue;
            tid_j += blockDim.x;
        }
        tid_i += blockDim.y;
        tid_j = threadIdx.x;
    }
}