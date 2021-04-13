# Parallel Computing

## Execute

`gcc -fopenmp -o main main.c && ./main`

## Function

    omp_get_thread_num(): get thread id
    omp_set_num_threads(n): set number of threads
    omp_get_num_threads(): get number of processing thread
    omp_set_schedule(): set schedule

## Command

    #pragma omp parallel
    #pragma omp for
    #pragma omp parallel sections
    #pragma omp section
    #pragma omp single
    #pragma omp master
    #pragma omp atomic
    #pragma omp parallel for
    #pragma omp parallel for private(j)​
    #pragma omp parallel for firstprivate(A) lastprivate(A)
    #pragma omp parallel for reduction(+:sum)

## Schedule

    1. static: 按照順序分配給thread(0~4)
    2. dynamic: 隨機分配給thread(誰有空誰去做)
        apply: load imbalanced
    3. guided: 資料量以遞減的方式交給thread執行
