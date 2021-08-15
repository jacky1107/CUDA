#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#define N 1000000
void random_numbers(int n, int a[n], int b[n], int c[n], int d[n]);
double cal_time(struct timespec *t_end, struct timespec *t_start);
void swap(int *p, int i, int j);

void serial_odd_even_sort(int *p, int size);
void openmp_odd_even_sort(int *p, int size);
void odd_even_sort_openmp_method1(int *p, int size);
void odd_even_sort_openmp_method2(int *p, int size);

int main()
{
    int i;
    int evaluate = 0;
    double final_time;
    struct timespec t_start, t_end;
    int *a = (int *)malloc(N * sizeof(int));
    int *b = (int *)malloc(N * sizeof(int));
    int *c = (int *)malloc(N * sizeof(int));
    int *d = (int *)malloc(N * sizeof(int));

    clock_gettime(CLOCK_REALTIME, &t_start);
    random_numbers(N, a, b, c, d);
    clock_gettime(CLOCK_REALTIME, &t_end);
    final_time = cal_time(&t_end, &t_start);
    printf("Generate random number time: %lf ms\n", final_time);

    // clock_gettime(CLOCK_REALTIME, &t_start);
    // serial_odd_even_sort(a, N);
    // clock_gettime(CLOCK_REALTIME, &t_end);
    // final_time = cal_time(&t_end, &t_start);
    // printf("Sequential time: %lf ms\n", final_time);

    // clock_gettime(CLOCK_REALTIME, &t_start);
    // odd_even_sort_openmp_method1(c, N);
    // clock_gettime(CLOCK_REALTIME, &t_end);
    // final_time = cal_time(&t_end, &t_start);
    // printf("Method 1 time: %lf ms\n", final_time);

    clock_gettime(CLOCK_REALTIME, &t_start);
    odd_even_sort_openmp_method2(d, N);
    clock_gettime(CLOCK_REALTIME, &t_end);
    final_time = cal_time(&t_end, &t_start);
    printf("Method 2 time: %lf ms\n", final_time);

    // clock_gettime(CLOCK_REALTIME, &t_start);
    // openmp_odd_even_sort(b, N);
    // clock_gettime(CLOCK_REALTIME, &t_end);
    // final_time = cal_time(&t_end, &t_start);
    // printf("Method 3 time: %lf ms\n", final_time);

    if (evaluate)
    {
        for (i = 0; i < N; i++)
        {
            if (a[i] != b[i] || a[i] != c[i] || a[i] != d[i])
            {
                break;
            }
        }
        if (i == N)
            printf("Test pass\n");
        else
            printf("Test failed\n");
    }
    return 0;
}

void random_numbers(int n, int a[], int b[], int c[], int d[])
{
    int i;
    for (i = 0; i < n; i++)
    {
        int value = rand() % N;
        a[i] = value;
        b[i] = value;
        c[i] = value;
        d[i] = value;
    }
}

void serial_odd_even_sort(int *p, int size)
{
    int swapped, i;
    do
    {
        swapped = 0;
        for (i = 1; i < size - 1; i += 2)
        {
            if (p[i] > p[i + 1])
            {
                swap(p, i, i + 1);
                swapped = 1;
            }
        }
        for (i = 0; i < size - 1; i += 2)
        {
            if (p[i] > p[i + 1])
            {
                swap(p, i, i + 1);
                swapped = 1;
            }
        }
    } while (swapped);
}

void openmp_odd_even_sort(int *p, int size)
{
    int swapped, i;
    do
    {
        swapped = 0;
#pragma omp sections
        {
#pragma omp section
            {
                for (i = 1; i < size - 1; i += 2)
                {
                    if (p[i] > p[i + 1])
                    {
                        swap(p, i, i + 1);
                        swapped = 1;
                    }
                }
            }
#pragma omp section
            {
                for (i = 0; i < size - 1; i += 2)
                {
                    if (p[i] > p[i + 1])
                    {
                        swap(p, i, i + 1);
                        swapped = 1;
                    }
                }
            }
        }
    } while (swapped);
}

void odd_even_sort_openmp_method1(int *a, int n)
{
    int phase, i, temp;
    for (phase = 0; phase < n; ++phase)
    {
        if (phase % 2 == 0)
        {
#pragma omp parallel for num_threads(16) default(none) shared(a, n) private(i, temp)
            for (i = 1; i < n; i += 2)
                if (a[i - 1] > a[i])
                {
                    swap(a, i, i - 1);
                }
        }
        else
        {
#pragma omp parallel for num_threads(16) default(none) shared(a, n) private(i, temp)
            for (i = 1; i < n - 1; i += 2)
                if (a[i] > a[i + 1])
                {
                    swap(a, i, i + 1);
                }
        }
    }
}

void odd_even_sort_openmp_method2(int *a, int n)
{
    int phase, i, temp;
#pragma omp parallel num_threads(16) default(none) shared(a, n) private(i, temp, phase)
    for (phase = 0; phase < n; ++phase)
    {
        if (phase % 2 == 0)
        {
#pragma omp for
            for (i = 1; i < n; i += 2)
                if (a[i - 1] > a[i])
                {
                    swap(a, i, i - 1);
                }
        }
        else
        {
#pragma omp for
            for (i = 1; i < n - 1; i += 2)
                if (a[i] > a[i + 1])
                {
                    swap(a, i, i + 1);
                }
        }
    }
}

void swap(int *p, int i, int j)
{
    int tmp = p[i];
    p[i] = p[j];
    p[j] = tmp;
}

double cal_time(struct timespec *t_end, struct timespec *t_start)
{
    double elapsedTime;
    elapsedTime = (t_end->tv_sec - t_start->tv_sec) * 1000.0;
    elapsedTime += (t_end->tv_nsec - t_start->tv_nsec) / 1000000.0;
    return elapsedTime;
}