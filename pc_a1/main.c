#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define N 2000
#define TIMES 20
int a[N][N], b[N][N];

int c[N][N];
int res[N][N];
int one_d[N * N];
int reduce[N][N];
int transpose_reduce[N][N];
int transpose_reduce_shared[N][N];

double eval_times[6][TIMES];

double cal_time(struct timespec *t_end, struct timespec *t_start)
{
    double elapsedTime;
    elapsedTime = (t_end->tv_sec - t_start->tv_sec) * 1000.0;
    elapsedTime += (t_end->tv_nsec - t_start->tv_nsec) / 1000000.0;
    return elapsedTime;
}

int test(int time)
{
    struct timespec t_start, t_end;
    int i, j, f, k;

    // Generate data
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            a[i][j] = rand() % 10;
            b[i][j] = rand() % 10;
        }

    // Sequential
    clock_gettime(CLOCK_REALTIME, &t_start);
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            c[i][j] = 0;
            for (k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];
        }
    clock_gettime(CLOCK_REALTIME, &t_end);
    double final_seq = cal_time(&t_end, &t_start);
    printf("Sequential time: %lf ms\n", final_seq);

    // Parallel
    clock_gettime(CLOCK_REALTIME, &t_start);
#pragma omp parallel for collapse(2) shared(res, a, b) schedule(dynamic, 16)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            res[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    clock_gettime(CLOCK_REALTIME, &t_end);
    double final_par = cal_time(&t_end, &t_start);
    printf("Parallel time: %lf ms\n", final_par);

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

    // Access 1D array + Reduce access times + Parallel
    clock_gettime(CLOCK_REALTIME, &t_start);
#pragma omp parallel for collapse(2) shared(one_d, a, b) schedule(dynamic, 16)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            one_d[i * N + j] = 0;
            for (int k = 0; k < N; k++)
            {
                sum += a[i][k] * b[k][j];
            }
            one_d[i * N + j] = sum;
        }
    }
    clock_gettime(CLOCK_REALTIME, &t_end);
    double final_1d = cal_time(&t_end, &t_start);
    printf("Parallel 1d time: %lf ms\n", final_1d);

    // Transpose + Reduce + parallel with no schedule
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
            transpose_reduce[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                transpose_reduce[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    clock_gettime(CLOCK_REALTIME, &t_end);
    double final_transpose_reduce = cal_time(&t_end, &t_start);
    printf("Parallel transpose_reduce time: %lf ms\n", final_transpose_reduce);

    for (int i = 0; i < N; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            int temp = b[i][j];
            b[i][j] = b[j][i];
            b[j][i] = temp;
        }
    }

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
            if (c[i][j] != res[i][j] || c[i][j] != one_d[i * N + j] || c[i][j] != reduce[i][j] || c[i][j] != transpose_reduce[i][j] || c[i][j] != transpose_reduce_shared[i][j])
            {
                break;
            }
        }
    }
    if (i == N && j == N)
        printf("Test pass!!!\n");
    else
    {
        printf("Test failure..\n");
        return 0;
    }

    eval_times[0][time] = final_seq;
    eval_times[1][time] = final_par;
    eval_times[2][time] = final_reduce;
    eval_times[3][time] = final_1d;
    eval_times[4][time] = final_transpose_reduce;
    eval_times[5][time] = best;
}

int main()
{
    for (int i = 0; i < TIMES; i++)
    {
        test(i);
    }

    double avg;
    for (int i = 0; i < 6; i++)
    {
        avg = 0;
        for (int j = 0; j < TIMES; j++)
        {
            avg += eval_times[i][j];
        }
        printf("Method %d: Avg=%lf\n", i, avg / TIMES);
    }

    return 0;
}
