# Parallel Computing

## Execute

`nvcc -o main main.cu && ./main`

## Function

- 1D

1. a = (int*)malloc( N * sizeof(int) ): 先 malloc cpu 記憶體
2. cudaMalloc( (void\*\*)&d*a, N * sizeof(int) ): 再 malloc cuda 記憶體
