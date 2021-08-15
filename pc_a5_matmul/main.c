#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define N 1024
int a[N][N], b[N][N];

int seq[N][N];
int reduce[N][N];
int transpose_reduce_shared[N][N];

double cal_time(struct timespec *t_end, struct timespec *t_start)
{
    double elapsedTime;
    elapsedTime = (t_end->tv_sec - t_start->tv_sec) * 1000.0;
    elapsedTime += (t_end->tv_nsec - t_start->tv_nsec) / 1000000.0;
    return elapsedTime;
}

int test()
{
    struct timespec t_start, t_end;
    int i, j, f, k;

    // Generate data
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            a[i][j] = rand() % N;
            b[i][j] = rand() % N;
        }

    // Sequential
    clock_gettime(CLOCK_REALTIME, &t_start);
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            seq[i][j] = 0;
            for (k = 0; k < N; k++)
                seq[i][j] += a[i][k] * b[k][j];
        }
    clock_gettime(CLOCK_REALTIME, &t_end);
    double final_seq = cal_time(&t_end, &t_start);
    printf("Sequential time: %lf ms\n", final_seq);

    // Reduce access times + Parallel
    clock_gettime(CLOCK_REALTIME, &t_start);
#pragma omp parallel for collapse(2) shared(reduce, a, b) schedule(dynamic, 16)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += a[i][k] * b[k][j];
            }
            reduce[i][j] = sum;
        }
    }
    clock_gettime(CLOCK_REALTIME, &t_end);
    double final_reduce = cal_time(&t_end, &t_start);
    printf("Parallel reduce time: %lf ms\n", final_reduce);

    // Transpose + Reduce + parallel
    clock_gettime(CLOCK_REALTIME, &t_start);
    for (int i = 0; i < N; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            int temp = b[i][j];
            b[i][j] = b[j][i];
            b[j][i] = temp;
        }
    }

#pragma omp parallel for shared(transpose_reduce_shared, a, b) collapse(2) schedule(dynamic, 16)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            transpose_reduce_shared[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                sum += a[i][k] * b[j][k];
            }
            transpose_reduce_shared[i][j] = sum;
        }
    }
    clock_gettime(CLOCK_REALTIME, &t_end);
    double best = cal_time(&t_end, &t_start);
    printf("Parallel transpose_reduce_shared time: %lf ms\n", best);

    // Evaluation
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (seq[i][j] != reduce[i][j] || seq[i][j] != transpose_reduce_shared[i][j])
            {
                break;
            }
        }
    }
    if (i == N && j == N)
    {
        printf("Test pass!!!\n");
    }
    else
    {
        printf("Test failure..\n");
        return 0;
    }

}

int main()
{
    test();
    return 0;
}
