#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define M 1024
#define K 1024
#define N 1024
#define SIZE 1024
#define NUM_THREADS 24

int A[M][K];
int B[K][N];
int C[M][N];
int goldenC[M][N];
struct v
{
    int i; /* row */
    int j; /* column */
};
void *worker(void *arg);

int main(int argc, char *argv[])
{
    int i, j, k;
    pthread_t tid[NUM_THREADS];       //Thread ID
    pthread_attr_t attr[NUM_THREADS]; //Set of thread attributes
    struct timespec t_start, t_end;
    double elapsedTime;

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }
    }

    // start time
    clock_gettime(CLOCK_REALTIME, &t_start);

    for (i = 0; i < NUM_THREADS; i++)
    {
        int *dataid;
        dataid = (int *)malloc(sizeof(int));
        *dataid = i;
        pthread_create(&tid[i], NULL, worker, (void *)dataid);
    }

    for (i = 0; i < NUM_THREADS; ++i)
    {
        pthread_join(tid[i], NULL);
    }
    // stop time
    clock_gettime(CLOCK_REALTIME, &t_end);

    // compute and print the elapsed time in millisec
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    printf("Parallel elapsedTime: %lf ms\n", elapsedTime);
    //Print out the resulting matrix

    // start time
    clock_gettime(CLOCK_REALTIME, &t_start);
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < K; k++)
            {
                goldenC[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    // stop time
    clock_gettime(CLOCK_REALTIME, &t_end);

    // compute and print the elapsed time in millisec
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    printf("Sequential elapsedTime: %lf ms\n", elapsedTime);

    int pass = 1;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (goldenC[i][j] != C[i][j])
            {
                pass = 0;
            }
        }
    }
    if (pass == 1)
        printf("Test pass!\n");

    return 0;
}

void *worker(void *arg)
{
    int i, j, k, tid, portion_size, row_start, row_end;
    double sum;

    tid = *(int *)(arg); // get the thread ID assigned sequentially.
    portion_size = SIZE / NUM_THREADS;
    row_start = tid * portion_size;
    row_end = (tid + 1) * portion_size;

    for (i = row_start; i < row_end; i++)
    { // hold row index of 'matrix1'
        for (j = 0; j < SIZE; j++)
        {            // hold column index of 'matrix2'
            sum = 0; // hold value of a cell
                     /* one pass to sum the multiplications of corresponding cells
	 in the row vector and column vector. */
            for (k = 0; k < SIZE; ++k)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}