#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 20000
#define TIMES 100

double cal_time(struct timespec *t_end, struct timespec *t_start)
{
  double elapsedTime;
  elapsedTime = (t_end->tv_sec - t_start->tv_sec) * 1000.0;
  elapsedTime += (t_end->tv_nsec - t_start->tv_nsec) / 1000000.0;
  return elapsedTime;
}

int par_vs_seq()
{
  int res;
  int i;
  float a[N], b[N], c[N], d[N];
  float cs[N], ds[N];
  struct timespec t_start, t_end;
  double final_seq;
  double final_par;

  for (i = 0; i < N; i++)
  {
    a[i] = i * 1.5;
    b[i] = i + 22.35;
  }
  // start time
  clock_gettime(CLOCK_REALTIME, &t_start);
#pragma omp parallel sections
  {
#pragma omp section
    {
#pragma omp parallel for
      for (int i = 0; i < N; i++)
      {
        c[i] = a[i] + b[i];
      }
#pragma omp parallel for
      for (int i = 0; i < N; i++)
      {
        d[i] = a[i] * b[i];
      }
    }
  }
  clock_gettime(CLOCK_REALTIME, &t_end);
  final_par = cal_time(&t_end, &t_start);
  printf("Parallel time: %lf ms\n", final_par);
  // stop time

  // Sequence
  clock_gettime(CLOCK_REALTIME, &t_start);
  for (i = 0; i < N; i++)
  {
    cs[i] = a[i] + b[i];
  }
  for (i = 0; i < N; i++)
    ds[i] = a[i] * b[i];
  clock_gettime(CLOCK_REALTIME, &t_end);
  final_seq = cal_time(&t_end, &t_start);
  printf("Sequential time: %lf ms\n", final_seq);

  // Evaluation
  for (i = 0; i < N; i++)
  {
    if (c[i] != cs[i] || d[i] != ds[i])
    {
      break;
    }
  }
  if (i == N)
  {
    printf("Test pass!!!\n");
    if (final_seq < final_par)
    {
      printf("Seq Wins\n");
      res = 0;
    }
    else
    {
      printf("Par Wins\n");
      res = 1;
    }
  }
  else
  {
    printf("Test failure..\n");
  }
  return res;
}

int main()
{
  int res;
  float total_parallel = 0;
  float total_sequence = 0;

  for (int i = 0; i < TIMES; i++)
  {
    res = par_vs_seq();
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
