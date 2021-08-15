#include <omp.h>
#include <stdio.h>
int main()
{
  int i, j;
  int s = 0;
  int N = 30;
  int A[N];
  int B[3];

  omp_set_num_threads(3);

  for (int j = 0; j < 3; j++)
  {
#pragma omp parallel for lastprivate(A)
    for (int i = 0; i < N / 3; i++)
    {
      // printf("%d\n", omp_get_thread_num());
      A[i] = i;
    }
    for (int k = 0; k < N; k++)
    {
      printf("%d\n", A[k]);
    }
  }

  return 0;
}
