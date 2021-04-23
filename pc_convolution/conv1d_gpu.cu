#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>


__global__ void conv( float *data, float *kernel, float *output ){
    int tid = threadIdx.x;
	int i;
	
	for(i = 0; i < 9; i++){
		output[tid] += data[tid + i] * kernel[i];
	}

}

int main(){
	int pass = 1;
	struct timespec t_start, t_end;
	double elapsedTimeCPU;
	
	float kernel[9] = {0.000229, 0.005977, 0.060598, 0.241732, 0.382928, 0.241732, 0.060598, 0.005977, 0.000229};
	float data[100] = {7.230, 16.98, 17.99, 1.703, 16.44, 4.484, 7.843, 13.44, 7.815, 11.91, 2.050, 6.138, 3.049, 0.167, 1.756, 10.46, 10.02, 10.48, 13.14, 7.329, 14.93, 7.275, 18.61, 13.82, 15.97, 11.43, 10.27, 5.290, 14.13, 2.671, 3.267, 6.149, 14.56, 13.11, 18.14, 16.47, 17.49, 16.20, 7.835, 5.883, 0.967, 0.237, 4.359, 13.15, 15.92, 16.94, 14.30, 17.47, 5.118, 5.142, 19.41, 5.046, 16.78, 3.944, 12.17, 7.983, 15.35, 7.839, 11.65, 12.56, 9.564, 14.30, 4.670, 1.893, 9.304, 0.173, 3.921, 15.63, 6.561, 16.25, 1.634, 4.870, 15.03, 0.269, 11.92, 0.390, 15.57, 2.918, 8.966, 14.04, 11.23, 7.519, 7.943, 6.570, 18.74, 15.54, 1.303, 14.01, 1.797, 1.526, 12.90, 3.051, 8.602, 7.094, 14.39, 14.13, 11.20, 2.637, 2.644, 2.810};
	float output[100-9+1];
	float output_from_device[100-9+1];
	
    // CPU
    clock_gettime( CLOCK_REALTIME, &t_start);
	for (int i = 0; i < 100-9+1; i++){
		output[i] = 0;
		for (int j = 0; j < 9; j++){
			output[i] += kernel[j] * data[i+j];
		}
	}
	clock_gettime( CLOCK_REALTIME, &t_end);
	elapsedTimeCPU = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTimeCPU += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("CPU elapsedTime: %lf ms\n", elapsedTimeCPU);
	
    // GPU
	float *d_kernel, *d_data, *d_output;
	cudaMalloc( (void**)&d_kernel, 9 * sizeof(float) );
	cudaMalloc( (void**)&d_data, 100 * sizeof(float) );
	cudaMalloc( (void**)&d_output, (100-9+1) * sizeof(float) );
	cudaMemcpy( d_kernel, kernel, 9 * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( d_data, data, 100 * sizeof(float), cudaMemcpyHostToDevice );
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
	conv<<<1, 100-9+1>>>(d_data, d_kernel, d_output);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);
	cudaMemcpy( output_from_device, d_output, (100-9+1) * sizeof(float), cudaMemcpyDeviceToHost );
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	
	//check correctness
	for (int i = 0; i < 100-9+1; i++){
		if(output_from_device[i]-output[i]>0.000001){ //don't use if(output_from_device[i]!=output[i])
			printf("CPU:%lf    GPU:%lf\n",output[i], output_from_device[i] );
			pass = 0;
		}
	}
	
	if(pass == 1)
		printf("Test pass!\n");
	else
		printf("Test fail!\n");
}