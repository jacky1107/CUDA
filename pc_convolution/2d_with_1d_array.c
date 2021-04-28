#include <stdio.h>
#include <math.h>
#include <stdbool.h> 
#include <sys/time.h>
#include <stdlib.h>
#define WIDTH1 4
#define WIDTH2 3

bool convolve2DSlow(unsigned char *in, unsigned char *out, int dataSizeX, int dataSizeY, double *kernel, int kernelSizeX, int kernelSizeY);

int main(){
    double elapsedTimeCPU;
    struct timespec t_start, t_end;

    unsigned char input[WIDTH1*WIDTH1] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    double Kernel[WIDTH2*WIDTH2] ={0,-1,0,-1,5,-1,0,-1,0};
    unsigned char output[WIDTH1*WIDTH1];
    int i, j;
    convolve2DSlow(input, output, WIDTH1, WIDTH1, Kernel, WIDTH2, WIDTH2);

    for(i=0; i<WIDTH1; i++){
        for(j=0; j<WIDTH1; j++){
            printf("%8d ", output[i*WIDTH1 + j]);
        }
        printf("\n");
    }
    return 0;
}
bool convolve2DSlow(unsigned char *in, unsigned char *out, int dataSizeX, int dataSizeY, double *kernel, int kernelSizeX, int kernelSizeY)
{
    int i, j, m, n, mm, nn;
    int kCenterX, kCenterY; // center index of kernel
    double sum; // temp accumulation buffer
    int rowIndex, colIndex;
    if(!in || !out || !kernel) return false;
    if(dataSizeX <= 0 || kernelSizeX <= 0) return false;
    kCenterX = kernelSizeX / 2;
    kCenterY = kernelSizeY / 2;
    for(i=0; i < dataSizeY; ++i){
        for(j=0; j < dataSizeX; ++j){
            sum = 0; // init to 0 before sum
            for(m=0; m < kernelSizeY; ++m){
                mm = kernelSizeY - 1 - m;
                for(n=0; n < kernelSizeX; ++n){
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