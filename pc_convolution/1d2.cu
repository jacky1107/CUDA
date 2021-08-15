#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#define N 1024 * 1024 * 128
#define KN 9
#define THREADSPERBLOCK 1
#define BLOCKSPERGRID (N + THREADSPERBLOCK - 1) / THREADSPERBLOCK

float data[N];
float kernel[KN];
float output[N-KN+1];
float output_from_device[N-KN+1];

__global__ void conv( float *data_cuda, float *kernel, float *output ){
    int tid = blockIdx.x;
    while (tid < N - KN + 1) {
        for(int i = 0; i < KN; i++) {
            output[tid] += data_cuda[tid + i] * kernel[i];
        }
        tid += gridDim.x;
    }
}

int main(){
    int cpu = true;
    int pass = 1;
    cudaError_t cuError = cudaSuccess ;

    double elapsedTimeCPU;
    struct timespec t_start, t_end;
    
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // generate dummy data
    srand(time(NULL));
    for (int i = 0; i < KN; i++) {
        kernel[i] = rand() / (float)RAND_MAX;
    }
    
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        data[i] = rand() / (float)RAND_MAX;
    }

    // CPU
    if (cpu) {
        clock_gettime( CLOCK_REALTIME, &t_start);
        for (int i = 0; i < N-KN+1; i++) {
            output[i] = 0;
            for (int j = 0; j < KN; j++) {
                output[i] += kernel[j] * data[i+j];
            }
        }
        clock_gettime( CLOCK_REALTIME, &t_end);
        elapsedTimeCPU = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
        elapsedTimeCPU += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
        printf("CPU elapsedTime: %lf ms\n", elapsedTimeCPU);
    }

    // GPU
    float *d_kernel, *d_data, *d_output;
    if (cudaMalloc( (void**)&d_kernel, KN * sizeof(float) ) != cudaSuccess) return 1;
    if (cudaMalloc( (void**)&d_data, N * sizeof(float) ) != cudaSuccess) return 1;
    if (cudaMalloc( (void**)&d_output, (N-KN+1) * sizeof(float) ) != cudaSuccess) return 1;
    if (cudaMemcpy( d_kernel, kernel, KN * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess) return 1;
    if (cudaMemcpy( d_data, data, N * sizeof(float), cudaMemcpyHostToDevice ) != cudaSuccess) return 1;

    cudaEventRecord(start, 0);
    conv<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(d_data, d_kernel, d_output);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);
    
    cudaMemcpy( output_from_device, d_output, (N-KN+1) * sizeof(float), cudaMemcpyDeviceToHost );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (cudaGetLastError() != cudaSuccess)
    {
        printf ("Failed in kernel launch and reason is %s\n", cudaGetErrorString(cuError)) ;
        return 1 ;
    }

    //check correctness
    if (cpu) {
        for (int i = 0; i < N-KN+1; i++){
            if((output_from_device[i] - output[i]) > 0.001){
                printf("CPU:%lf GPU:%lf\n",output[i], output_from_device[i] );
                pass = 0;
                break;
            }
        }
        if(pass == 1) {
            printf("Test pass!\n");
            printf("GPU / CPU = %f\n", elapsedTimeCPU / elapsedTime);
        }
        else
            printf("Test fail!\n");
    }
}