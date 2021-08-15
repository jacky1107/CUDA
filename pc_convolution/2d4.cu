#include <stdio.h>
#include <math.h>
#include <stdbool.h> 
#include <cuda_runtime.h>
#define H 10000
#define W 10000
#define K 3
#define N H * W
#define THREADSPERBLOCK 32
#define BLOCKSPERGRIDX 1
#define BLOCKSPERGRIDY 1

double kernel[K * K];
unsigned char input[N];
unsigned char output[N];
unsigned char outputGPU[N];

bool convolve2DSlow(unsigned char *in, unsigned char *out, int dataSizeX, int dataSizeY, double *kernel, int kernelSizeX, int kernelSizeY);
__global__ void convolve2D_GPU( unsigned char *input, unsigned char *output, double *kernel);

int main() {
    int cpu = true;

    double elapsedTimeCPU;
    struct timespec t_start, t_end;

    float elapsedTimeGPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int i, j;

    srand(time(NULL));
    for (i = 0; i < N; i++) {
        input[i] = (unsigned char) (rand() % 255 + 1);
    }

    srand(time(NULL));
    for (i = 0; i < K * K; i++) {
        kernel[i] = (unsigned char) (rand() % 255 + 1);
    }

    double *d_kernel;
    unsigned char *d_input;
    unsigned char *d_output;

    if (cpu) {
        clock_gettime( CLOCK_REALTIME, &t_start);
        convolve2DSlow(input, output, W, H, kernel, K, K);
        clock_gettime( CLOCK_REALTIME, &t_end);
        elapsedTimeCPU = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
        elapsedTimeCPU += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
        printf("CPU elapsedTime: %lf ms\n", elapsedTimeCPU);
    }

    cudaMalloc( (void**)&d_input, N * sizeof(unsigned char) );
    cudaMalloc( (void**)&d_kernel, K * K * sizeof(double) );
    cudaMalloc( (void**)&d_output, N * sizeof(unsigned char) );
    cudaMemcpy( d_input, input, N * sizeof(unsigned char), cudaMemcpyHostToDevice );
    cudaMemcpy( d_kernel, kernel, K * K * sizeof(double), cudaMemcpyHostToDevice );

    dim3 dimGrid (BLOCKSPERGRIDX, BLOCKSPERGRIDY, 1);
    dim3 dimBlock (THREADSPERBLOCK, THREADSPERBLOCK, 1);

    cudaEventRecord(start, 0);
    convolve2D_GPU<<<dimGrid, dimBlock>>>(d_input, d_output, d_kernel);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&elapsedTimeGPU, start, stop);
    printf("GPU time: %13f msec\n", elapsedTimeGPU);
    cudaMemcpy( outputGPU, d_output, N * sizeof(unsigned char), cudaMemcpyDeviceToHost );

    //check results
    if (cpu) {
        int pass = 1;
        for(i = 0; i < H; i++){
            for(j = 0; j < W; j++){
                if((output[i * W + j] - outputGPU[i * W + j]) > 0.00001){
                    pass = 0;
                    break;
                }
            }
        }

        if(pass == 0) printf("Test Fail!\n");
        else{
            printf("Test pass!\n");
            printf("GPU / CPU = %f\n", elapsedTimeCPU / elapsedTimeGPU);
        }
    }
    return 0;
}

__global__ void convolve2D_GPU( unsigned char *input, unsigned char *output, double *kernel) {
    int m, n, mm, nn;
    int kCenterX, kCenterY;

    double sum;

    int rowIndex, colIndex;
    int dataSizeX, dataSizeY, kernelSizeX, kernelSizeY;

    dataSizeX = W;
    dataSizeY = H;
    kernelSizeX = kernelSizeY = K;

    kCenterX = kernelSizeX / 2;
    kCenterY = kernelSizeY / 2;

    int x, y;
    int tx = threadIdx.x;
    int ty = blockDim.x * threadIdx.y;
    int tid = tx + ty;
    while (tid < H * W) {
        sum = 0;
        x = tid % W;
        y = tid / W;
        for(m = 0; m < kernelSizeY; ++m){
            mm = kernelSizeY - 1 - m;
            for(n = 0; n < kernelSizeX; ++n){
                nn = kernelSizeX - 1 - n; 
                rowIndex = y + (kCenterY - mm);
                colIndex = x + (kCenterX - nn);
                if(rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
                    sum += input[dataSizeX * rowIndex + colIndex] * kernel[kernelSizeX * m + n];
            }
        }
        output[tid] = (unsigned char)(fabs(sum) + 0.5f);
        tid = tid + blockDim.x * blockDim.y;
    }
}

bool convolve2DSlow(unsigned char *in, unsigned char *out, int dataSizeX, int dataSizeY, double *kernel, int kernelSizeX, int kernelSizeY) {
    int i, j, m, n, mm, nn;
    int kCenterX, kCenterY;
    double sum;
    int rowIndex, colIndex;
    if(!in || !out || !kernel) return false;
    if(dataSizeX <= 0 || kernelSizeX <= 0) return false;
    kCenterX = kernelSizeX / 2;
    kCenterY = kernelSizeY / 2;
    for(i = 0; i < dataSizeY; ++i){
        for(j = 0; j < dataSizeX; ++j){
            sum = 0;
            for(m = 0; m < kernelSizeY; ++m){
                mm = kernelSizeY - 1 - m;
                for(n = 0; n < kernelSizeX; ++n){
                    nn = kernelSizeX - 1 - n; 
                    rowIndex = i + (kCenterY - mm);
                    colIndex = j + (kCenterX - nn);
                    if(rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
                        sum += in[dataSizeX * rowIndex + colIndex] * kernel[kernelSizeX * m + n];
                }
            }
            out[dataSizeX * i + j] = (unsigned char)(fabs(sum) + 0.5f);
        }
    }
    return true;
}

