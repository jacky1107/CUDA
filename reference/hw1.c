#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define N 1000
#define TIMES 1
int a[N][N], b[N][N], one_d[N * N], reduce[N][N], transpose_reduce[N][N];

// int s1[N][N], s2[N][N];

// int p1;
// int p2;
// int p3;
// int p4;
// int p5;
// int p6;
// int p7;

// p1 = (s1[0][0] + s1[1][1]) * (s2[0][0] + s2[1][1]);
// p2 = (s1[1][0] + s1[1][1]) * s2[0][0];
// p3 = s1[0][0] * (s2[0][1] - s2[1][1]);
// p4 = s1[1][1] * (s2[1][0] - s2[0][0]);
// p5 = (s1[0][0] + s1[0][1]) * s2[1][1];
// p6 = (s1[1][0] - s1[0][0]) * (s2[0][0] + s2[0][1]);
// p7 = (s1[0][1] - s1[1][1]) * (s2[1][0] + s2[1][1]);

// divide[0][0] = p1 + p4 - p5 + p7;
// divide[0][1] = p3 + p5;
// divide[1][0] = p2 + p4;
// divide[1][1] = p1 + p3 - p2 + p6;

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
  int c[N][N];
  int res[N][N];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
    {
      a[i][j] = rand() % 10;
      b[i][j] = rand() % 10;
    }

  clock_gettime(CLOCK_REALTIME, &t_start);
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
    {
      c[i][j] = 0;
      for (k = 0; k < N; k++)
        c[i][j] += a[i][k] * b[k][j];
    }
  clock_gettime(CLOCK_REALTIME, &t_end);
  float final_seq = cal_time(&t_end, &t_start);
  printf("Sequential time: %lf ms\n", final_seq);

  clock_gettime(CLOCK_REALTIME, &t_start);
#pragma omp parallel for collapse(2) shared(res, a, b)
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
  float final_par = cal_time(&t_end, &t_start);
  printf("Parallel time: %lf ms\n", final_par);

  clock_gettime(CLOCK_REALTIME, &t_start);
#pragma omp parallel for collapse(2) shared(one_d, a, b)
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      float sum = 0;
      one_d[i * N + j] = 0;
      for (int k = 0; k < N; k++)
      {
        sum += a[i][k] * b[k][j];
      }
      one_d[i * N + j] = sum;
    }
  }
  clock_gettime(CLOCK_REALTIME, &t_end);
  float final_1d = cal_time(&t_end, &t_start);
  printf("Parallel reduce + 1d time: %lf ms\n", final_1d);

  // Divide and conquer best
  clock_gettime(CLOCK_REALTIME, &t_start);
#pragma omp parallel for collapse(2) shared(reduce, a, b)
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      float sum = 0;
      for (int k = 0; k < N; k++)
      {
        sum += a[i][k] * b[k][j];
      }
      reduce[i][j] = sum;
    }
  }

  clock_gettime(CLOCK_REALTIME, &t_end);
  float final_reduce = cal_time(&t_end, &t_start);
  printf("Parallel reduce time: %lf ms\n", final_reduce);

  //
  clock_gettime(CLOCK_REALTIME, &t_start);

  for (int i = 0; i < N; i++)
  {
    for (int j = i + 1; j < N; j++)
    {
      float temp = b[i][j];
      b[i][j] = b[j][i];
      b[j][i] = temp;
    }
  }

#pragma omp parallel for collapse(2) shared(transpose_reduce, a, b)
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      float sum = 0;
      transpose_reduce[i][j] = 0;
      for (int k = 0; k < N; k++)
      {
        sum += a[i][k] * b[j][k];
      }
      transpose_reduce[i][j] = sum;
    }
  }

  clock_gettime(CLOCK_REALTIME, &t_end);
  float final_transpose_reduce = cal_time(&t_end, &t_start);
  printf("Parallel transpose_reduce time: %lf ms\n", final_transpose_reduce);

  // Evaluation
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      if (c[i][j] != res[i][j] || c[i][j] != one_d[i * N + j] || c[i][j] != reduce[i][j] || c[i][j] != transpose_reduce[i][j])
      {
        break;
      }
    }
  }
  if (i == N && j == N)
  {
    printf("Test pass!!!\n");
    return final_seq > final_par;
  }
  else
    // printf("Test failure..\n");
    return -1;
}

int main()
{
  int res;
  float total_parallel = 0;
  float total_sequence = 0;
  for (int i = 0; i < TIMES; i++)
  {
    res = test();
    if (res == 0)
    {
      total_sequence += 1;
    }
    else
    {
      total_parallel += 1;
    }
  }

  float ratio = (total_parallel / total_sequence);
  printf("Total of Parallel: %f\n", total_parallel);
  printf("Total of Sequence: %f\n", total_sequence);
  printf("Parallel / Sequence = %f\n", ratio);
  return 0;
}
