#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define M 1000
#define K 1000
#define N 1000
#define SIZE 1000
#define NUM_THREADS 16
#define TILE_WIDTH 250

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
    pthread_t tid[SIZE / TILE_WIDTH][SIZE / TILE_WIDTH];       //Thread ID
    pthread_attr_t attr[SIZE / TILE_WIDTH][SIZE / TILE_WIDTH]; //Set of thread attributes
    struct timespec t_start, t_end;
    double elapsedTime;

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i][j] = rand() % SIZE;
            B[i][j] = rand() % SIZE;
        }
    }

    // start time
    clock_gettime(CLOCK_REALTIME, &t_start);

    for (i = 0; i < SIZE / TILE_WIDTH; i++)
    {
        for (j = 0; j < SIZE / TILE_WIDTH; j++)
        {
            struct v *dataid = (struct v *)malloc(sizeof(struct v));
            dataid->i = i;
            dataid->j = j;
            pthread_create(&tid[i][j], NULL, worker, (void *)dataid);
            //printf("%d %d\n", dataid->i, dataid->j);
        }
    }

    for (i = 0; i < SIZE / TILE_WIDTH; ++i)
    {
        for (j = 0; j < SIZE / TILE_WIDTH; j++)
        {
            pthread_join(tid[i][j], NULL);
        }
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
                printf("%d %d\n", i, j);
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
    int i, j, k, tid, portion_size, row_start, row_end, column_start, column_end;
    double sum;
    struct v *data = (struct v *)arg;
    //tid = *(int *)(arg); // get the thread ID assigned sequentially.
    portion_size = TILE_WIDTH;
    row_start = data->i * portion_size;
    row_end = (data->i + 1) * portion_size;
    column_start = data->j * portion_size;
    column_end = (data->j + 1) * portion_size;
    //printf("%d %d %d %d %d %d\n", data->i, data->j, row_start, row_end, column_start, column_end);
    for (i = row_start; i < row_end; i++)
    { // hold row index of 'matrix1'
        for (j = column_start; j < column_end; j++)
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