#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 10000000

double cal_time(struct timespec *t_end, struct timespec *t_start)
{
  double elapsedTime;
  elapsedTime = (t_end->tv_sec - t_start->tv_sec) * 1000.0;
  elapsedTime += (t_end->tv_nsec - t_start->tv_nsec) / 1000000.0;
  return elapsedTime;
}

int main()
{
  struct timespec t_start, t_end;
  int i;
  int temp;
  int *A, *B, *AA, *BB;
  A = (int *)malloc(N * sizeof(int));
  B = (int *)malloc(N * sizeof(int));
  AA = (int *)malloc(N * sizeof(int));
  BB = (int *)malloc(N * sizeof(int));
  for (i = 0; i < N; i++)
  {
    A[i] = rand() % 256;
    B[i] = rand() % 256;
    AA[i] = A[i];
    BB[i] = B[i];
  }

  clock_gettime(CLOCK_REALTIME, &t_start);
  for (i = 0; i < N; i++)
  {
    temp = A[i];
    A[i] = B[i];
    B[i] = temp;
  }
  clock_gettime(CLOCK_REALTIME, &t_end);
  printf("Sequential time: %lf ms\n", cal_time(&t_end, &t_start));

  clock_gettime(CLOCK_REALTIME, &t_start);
#pragma omp parallel for
  for (i = 0; i < N; i++)
  {
    int temp; // Need to init every times
    temp = AA[i];
    AA[i] = BB[i];
    BB[i] = temp;
  }
  clock_gettime(CLOCK_REALTIME, &t_end);
  printf("Parallel time: %lf ms\n", cal_time(&t_end, &t_start));

  for (i = 0; i < N; i++)
  {
    if (A[i] != AA[i] || B[i] != BB[i])
      break;
  }
  if (i == N)
    printf("Test pass!!!\n");
  else
    printf("Test failure\n");
  return 0;
}
