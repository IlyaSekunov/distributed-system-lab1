#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct point {
    double x, y;
} point;

int n;
int comm_sz, my_rank;

point GetRandomPoint() {
    point p;
    p.x = drand48() * 2.0;
    p.y = drand48() * 2.0;
    return p;
}

int InCircle(point p) {
    return (p.x - 1) * (p.x - 1) + (p.y - 1) * (p.y - 1) <= 1;
}

int main() {
    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        printf("Введите кол-во попыток: ");
        fflush(stdout);
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    srand48(time(NULL) + my_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    int local_n = (n + comm_sz - 1) / comm_sz;
    int local_in = 0;
    for (int i = 0; i < local_n; ++i) {
        point p = GetRandomPoint();
        local_in += InCircle(p);
    }

    int global_in = 0;
    MPI_Reduce(&local_in, &global_in, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    double local_time = end - start;
    double global_time = 0;
    MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        double pi = global_in * 4. / n;
        printf("Число Пи: %lf\n", pi);
        printf("Время исполнения: %lf мс\n", global_time * 1000);
    }

    MPI_Finalize();

    return 0;
}