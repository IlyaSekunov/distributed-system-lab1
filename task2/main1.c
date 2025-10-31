#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_NUM 10000

void GenerateMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() % MAX_NUM;
    }
}

void PrintMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void InputMatrix(int my_rank, int *n, int *m) {
    if (my_rank == 0) {
        printf("Введите размеры матрицы n * m: ");
        fflush(stdout);
        scanf("%d%d", n, m);
    }
    MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

int main() {
    MPI_Init(NULL, NULL);

    int my_rank, comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    int n, m;
    InputMatrix(my_rank, &n, &m);

    int *matrix = NULL;
    int *vector = calloc(m, sizeof(int));
    if (my_rank == 0) {
        matrix = calloc(n * m, sizeof(int));
        GenerateMatrix(matrix, n, m);
        GenerateMatrix(vector, 1, m);
    }

    // Начало замера времени - момент, когда матрица и вектор сгенерированы
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Bcast(vector, m, MPI_INT, 0, MPI_COMM_WORLD);
    
    int rows_per_process = n / comm_sz;
    int remainder = n % comm_sz;

    int *sendcounts = NULL, *displs = NULL;
    if (my_rank == 0) {
        sendcounts = calloc(comm_sz, sizeof(int));
        displs = calloc(comm_sz, sizeof(int));
        for (int i = 0; i < comm_sz; ++i) {
            sendcounts[i] = (rows_per_process + (i < remainder)) * m;
            displs[i] = (i ? displs[i - 1] : 0) + (i ? sendcounts[i - 1] : 0);
        }
    }

    int local_n = rows_per_process + (my_rank < remainder);
    int* local_matrix = calloc(local_n * m, sizeof(int));
    MPI_Scatterv(
        matrix,
        sendcounts,
        displs,
        MPI_INT,
        local_matrix,
        local_n * m,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );
    
    int *local_res = calloc(local_n, sizeof(int));
    for (int i = 0; i < local_n; ++i) {
        local_res[i] = 0;
        for (int j = 0; j < m; ++j) {
            local_res[i] += local_matrix[i * m + j] * vector[j];
        }
    }

    int *global_res = NULL;
    if (my_rank == 0) {
        global_res = calloc(n, sizeof(int));
        for (int i = 0; i < comm_sz; ++i) {
            sendcounts[i] = rows_per_process + (i < remainder);
            displs[i] = (i ? displs[i - 1] : 0) + (i ? sendcounts[i - 1] : 0); 
        }
    }

    MPI_Gatherv(
        local_res,
        local_n,
        MPI_INT,
        global_res,
        sendcounts,
        displs,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    double end = MPI_Wtime();
    double time = end - start;
    if (my_rank == 0) {
        printf("Время исполнения: %lf мс\n", time * 1000);

        free(global_res);
        free(sendcounts);
        free(displs);
    }

    free(local_matrix);
    free(vector);
    free(local_res);
    
    MPI_Finalize();
    return 0;
}