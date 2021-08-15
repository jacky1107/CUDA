#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 1000
int s[N], res[N];

double cal_time(struct timespec *t_end, struct timespec *t_start)
{
  double elapsedTime;
  elapsedTime = (t_end->tv_sec - t_start->tv_sec) * 1000.0;
  elapsedTime += (t_end->tv_nsec - t_start->tv_nsec) / 1000000.0;
  return elapsedTime;
}

void serial_odd_even_sort(int *p, int size);
void parallel_odd_even_sort(int *p, int size);
int main()
{
  int i;
  struct timespec t_start, t_end;

  for (int i = 0; i < N; i++)
  {
    int a = (rand() % N) + 1;
    s[i] = a;
    res[i] = a;
  }

  clock_gettime(CLOCK_REALTIME, &t_start);
  serial_odd_even_sort(s, N);
  clock_gettime(CLOCK_REALTIME, &t_end);
  float final_seq = cal_time(&t_end, &t_start);
  printf("Sequential time: %lf ms\n", final_seq);

  clock_gettime(CLOCK_REALTIME, &t_start);
  parallel_odd_even_sort(res, N);
  clock_gettime(CLOCK_REALTIME, &t_end);
  float final_parallel = cal_time(&t_end, &t_start);
  printf("Parallel time: %lf ms\n", final_parallel);

  for (i = 0; i < N; i++)
  {
    if (s[i] != res[i])
    {
      break;
    }
  }

  if (i == N)
    printf("Test Pass\n");
  else
    printf("Test Failed\n");
}

void parallel_odd_even_sort(int *p, int size)
{
  int swapped, i;
  int temp;
  do
  {
    swapped = 0;

#pragma omp parallel for private(temp, i)
    for (i = 1; i < size - 1; i += 2)
    { //odd
      if (p[i] > p[i + 1])
      {
        temp = p[i];
        p[i] = p[i + 1];
        p[i + 1] = temp;
        swapped = 1;
      }
    }

#pragma omp parallel for private(temp, i)
    for (i = 0; i < size - 1; i += 2)
    { //even
      if (p[i] > p[i + 1])
      {
        temp = p[i];
        p[i] = p[i + 1];
        p[i + 1] = temp;
        swapped = 1;
      }
    }

  } while (swapped);
}

void serial_odd_even_sort(int *p, int size)
{
  int swapped, i;
  int temp;
  do
  {
    swapped = 0;

    for (i = 1; i < size - 1; i += 2)
    { //odd
      if (p[i] > p[i + 1])
      {
        temp = p[i];
        p[i] = p[i + 1];
        p[i + 1] = temp;
        swapped = 1;
      }
    }

    for (i = 0; i < size - 1; i += 2)
    { //even
      if (p[i] > p[i + 1])
      {
        temp = p[i];
        p[i] = p[i + 1];
        p[i + 1] = temp;
        swapped = 1;
      }
    }

  } while (swapped);
}