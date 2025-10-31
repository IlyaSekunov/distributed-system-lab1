#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

void PrintMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void initialize_data(int* matrix, int* vector, int rows, int cols) {
    srand(time(NULL));
    for (int i = 0; i < rows * cols; i++) matrix[i] = (int)(rand() % 100) / 10.0; 
    for (int i = 0; i < cols; i++) vector[i] = (int)(rand() % 100) / 10.0; 

    // printf("Матрица: \n");
    // PrintMatrix(matrix, rows, cols);
    // printf("Вектор: \n");
    // PrintMatrix(vector, 1, cols);
}

void matrix_vector_multiply_block(int* local_matrix, int* vector, 
                                 int block_rows, int block_cols,
                                 int total_rows, int total_cols,
                                 MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    MPI_Bcast(vector, total_cols, MPI_INT, 0, comm);
    
    int *local_result = (int*)malloc(block_rows*sizeof(int));
    int blocks_in_row = total_cols / block_cols;
    for (int i = 0; i < block_rows; i++) {
        local_result[i] = 0.0;
        int oj = (rank%blocks_in_row)*(block_cols);
        for (int j = 0; j < block_cols; j++)
            local_result[i] += local_matrix[i*block_cols+j] * vector[oj + j];
    }

    int *global_matrix = NULL;
    if (rank == 0) global_matrix = (int*)malloc(size*block_rows*sizeof(int));

    MPI_Gather(local_result, block_rows, MPI_INT,
               global_matrix, block_rows, MPI_INT,
               0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int *final_matrix = (int*)calloc(total_rows, sizeof(int));    

        for (int bi = 0, idx = 0; bi < size; bi++) {
            int oi = bi/blocks_in_row*block_rows;
            for (int i = 0; i < block_rows; i++, idx++) {
                final_matrix[i+oi] += global_matrix[idx];
            }
        }

        // printf("Итоговый вектор: \n");
        // PrintMatrix(final_matrix, 1, total_rows);
    }
}

void run_experiments(int n, int m, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    int rows = n;
    int cols = m;

    int block_rows = (int)sqrt(rows * cols / size);
    int block_cols = (int)sqrt(cols * rows / size);

    int blocks_in_row = cols / block_cols;
    int blocks_in_col = rows / block_rows;
    
    int *matrix = NULL, *vector = NULL;
    if (rank == 0) {
        matrix = (int*)malloc(rows * cols * sizeof(int));
        vector = (int*)malloc(cols * sizeof(int));
        initialize_data(matrix, vector, rows, cols);
    } else vector = (int*)malloc(cols * sizeof(int));
    
    int* scatter_matrix = (int*)malloc(rows * cols * sizeof(int));
    if (rank == 0)
        for (int bi = 0, idx = 0; bi < size; bi++) {
            int oi = bi/blocks_in_row * block_rows;
            int oj = bi%blocks_in_row * block_cols;
            for (int i = 0; i < block_rows; i++)
                for (int j = 0; j < block_cols; j++, idx++)
                    scatter_matrix[idx] = matrix[(oi+i)*cols + (j+oj)];
        }

    
    double start_time = MPI_Wtime();

    int* local_matrix_block = (int*)malloc(block_rows * block_cols * sizeof(int));
    MPI_Scatter(scatter_matrix, block_cols*block_rows, MPI_INT,
                local_matrix_block, block_cols*block_rows, MPI_INT,
                0, MPI_COMM_WORLD);
    matrix_vector_multiply_block(local_matrix_block, vector, block_rows, block_cols, rows, cols, comm);

    double time_block = MPI_Wtime() - start_time;
    
    if (rank == 0) printf("%lf\n", time_block * 1000); 
    
    free(local_matrix_block);
    if (rank == 0) free(matrix);
    free(vector);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int n = 10000, m = 10000;

    run_experiments(n, m, MPI_COMM_WORLD);
    
    MPI_Finalize();
    return 0;
}
