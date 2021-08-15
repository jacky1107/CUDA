#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 500000000

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
  double elapsedTime;
  int i;
  int *A, *B, *C, *CC;
  A = (int *)malloc(N * sizeof(int));
  B = (int *)malloc(N * sizeof(int));
  C = (int *)malloc(N * sizeof(int));
  CC = (int *)malloc(N * sizeof(int));
  for (i = 0; i < N; i++)
  {
    A[i] = rand() % 256;
    B[i] = rand() % 256;
  }
  // start time
  clock_gettime(CLOCK_REALTIME, &t_start);
  for (i = 0; i < N; i++)
  {
    C[i] = A[i] + B[i];
  }
  // stop time
  clock_gettime(CLOCK_REALTIME, &t_end);
  printf("Sequential time: %lf ms\n", cal_time(&t_end, &t_start));

  // start time
  clock_gettime(CLOCK_REALTIME, &t_start);

// if u use this line without parallel, it onle use master thread
#pragma omp parallel for
  for (i = 0; i < N; i++)
  {
    CC[i] = A[i] + B[i];
  }
  // stop time
  clock_gettime(CLOCK_REALTIME, &t_end);
  printf("Parallel elapsedTime: %lf ms\n", cal_time(&t_end, &t_start));

  // Calculate time
  for (i = 0; i < N; i++)
  {
    if (CC[i] != C[i])
      break;
  }
  if (i == N)
    printf("Test pass!!!\n");

  return 0;
}