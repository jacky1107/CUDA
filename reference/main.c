#include <omp.h>
#include <stdio.h>
int main()
{
  int i, j;
  int s = 0;
  int N = 20000;

  double start = omp_get_wtime();

  ////////////////
  // private(j) //
  ////////////////
  // #pragma omp for private(j)
  //   for (int i = 0; i < N; i++) {
  //     for (j = 0; j < N; j++) {
  //       s += 1;
  //     }
  //   }
  //   printf("%d\n", s);
  //   printf("Time: %4fs\n", omp_get_wtime() - start);

  ////////////
  // atomic //
  ////////////
  //   start = omp_get_wtime();
  //   s = 0;
  // #pragma omp parallel for
  //   for (i = 0; i < N; i++) {
  //     for (int j = 0; j < N; j++)
  // #pragma omp atomic
  //       s += 1;
  //   }
  //   printf("%d\n", s);
  //   printf("Time: %4fs\n", omp_get_wtime() - start);

  //////////////////////////
  // reduction(+ : param) //
  //////////////////////////
  //   s = 0;
  //   start = omp_get_wtime();
  // #pragma omp parallel for reduction(+ : s)
  //   for (i = 0; i < N; i++) {
  //     for (int j = 0; j < N; j++)
  //       s += 1;
  //   }
  //   printf("%d\n", s);
  //   printf("Time: %4fs\n", omp_get_wtime() - start);

  /////////////////////////
  // schedule(static, 4) //
  /////////////////////////
  //   s = 0;
  //   start = omp_get_wtime();
  // #pragma omp parallel for schedule(static, 4) num_threads(2) ordered
  //   for (int i = 0; i < 16; i++) {
  // #pragma omp ordered
  //     printf("Thread %d has completed iteration %d\n", omp_get_thread_num(),
  //     i);
  //   }
  return 0;
}
