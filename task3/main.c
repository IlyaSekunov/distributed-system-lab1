#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MATRIX_SIZE 1444
#define MAX_NUM 10

void GenerateMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() % (MAX_NUM + 1);
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

int main() {
    MPI_Init(NULL, NULL);

    int my_rank, comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    int q = (int)sqrt(comm_sz);
    int n = MATRIX_SIZE;
    int block_size = n / q;
    
    int *local_A = calloc(block_size * block_size, sizeof(int));
    int *local_B = calloc(block_size * block_size, sizeof(int));
    int *local_C = calloc(block_size * block_size, sizeof(int));
    
    MPI_Comm grid_comm;
    int dims[2] = {q, q};
    int periods[2] = {1, 1};
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);
    
    int coords[2];
    MPI_Cart_coords(grid_comm, my_rank, 2, coords);
    int row = coords[0];
    int col = coords[1];
    
    int *A_full = NULL;
    int *B_full = NULL;
    int *C_full = NULL;
    if (my_rank == 0) {
        A_full = calloc(n * n, sizeof(int));
        B_full = calloc(n * n, sizeof(int));
        C_full = calloc(n * n, sizeof(int));
        
        srand(time(NULL));
        GenerateMatrix(A_full, n, n);
        GenerateMatrix(B_full, n, n);
    }
    
    MPI_Datatype block_type;
    MPI_Type_vector(block_size, block_size, n, MPI_INT, &block_type);
    MPI_Type_commit(&block_type);
    
    if (my_rank == 0) {
        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                int dest_rank;
                int dest_coords[2] = {i, j};
                MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);
                
                if (dest_rank == 0) {
                    for (int x = 0; x < block_size; x++) {
                        for (int y = 0; y < block_size; y++) {
                            local_A[x * block_size + y] = A_full[(i * block_size + x) * n + (j * block_size + y)];
                            local_B[x * block_size + y] = B_full[(i * block_size + x) * n + (j * block_size + y)];
                        }
                    }
                } else {
                    MPI_Send(
                        &A_full[i * block_size * n + j * block_size], 
                        1, 
                        block_type, 
                        dest_rank, 
                        0, 
                        grid_comm
                    );
                    MPI_Send(
                        &B_full[i * block_size * n + j * block_size], 
                        1, 
                        block_type, 
                        dest_rank, 
                        1, 
                        grid_comm
                    );
                }
            }
        }
    } else {
        MPI_Recv(local_A, block_size * block_size, MPI_INT, 0, 0, grid_comm, MPI_STATUS_IGNORE);
        MPI_Recv(local_B, block_size * block_size, MPI_INT, 0, 1, grid_comm, MPI_STATUS_IGNORE);
    }
    
    MPI_Status status;
    int left, right, up, down;
    
    MPI_Cart_shift(grid_comm, 1, -1, &left, &right);
    MPI_Cart_shift(grid_comm, 0, -1, &up, &down);
    
    int *temp_A = calloc(block_size * block_size, sizeof(int));
    memcpy(temp_A, local_A, block_size * block_size * sizeof(int));
    
    MPI_Sendrecv_replace(
        temp_A, 
        block_size * block_size, 
        MPI_INT, 
        left, 
        0, 
        right, 
        0, 
        grid_comm, 
        &status
    );
    
    int *temp_B = calloc(block_size * block_size, sizeof(int));
    memcpy(temp_B, local_B, block_size * block_size * sizeof(int));
    
    MPI_Sendrecv_replace(
        temp_B, 
        block_size * block_size, 
        MPI_INT, 
        up, 
        0, 
        down, 
        0, 
        grid_comm, 
        &status
    );
    
    for (int step = 0; step < q; step++) {
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    local_C[i * block_size + j] += temp_A[i * block_size + k] * temp_B[k * block_size + j];
                }
            }
        }
        
        // Сдвиг влево
        MPI_Sendrecv_replace(
            temp_A, 
            block_size * block_size, 
            MPI_INT, 
            left, 
            0, 
            right, 
            0, 
            grid_comm, 
            &status
        );
        
        // Сдвиг вверх
        MPI_Sendrecv_replace(
            temp_B, 
            block_size * block_size, 
            MPI_INT, 
            up, 
            0, 
            down, 
            0, 
            grid_comm, 
            &status
        );
    }
    
    free(temp_A);
    free(temp_B);
    
    if (my_rank == 0) {
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                C_full[i * n + j] = local_C[i * block_size + j];
            }
        }
        
        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                if (i == 0 && j == 0) continue;
                
                int source_rank;
                int source_coords[2] = {i, j};
                MPI_Cart_rank(grid_comm, source_coords, &source_rank);
                
                int *recv_buffer = calloc(block_size * block_size, sizeof(int));
                MPI_Recv(
                    recv_buffer, 
                    block_size * block_size, 
                    MPI_INT, 
                    source_rank, 
                    0, 
                    grid_comm, 
                    MPI_STATUS_IGNORE
                );
                
                for (int x = 0; x < block_size; x++) {
                    for (int y = 0; y < block_size; y++) {
                        C_full[(i * block_size + x) * n + (j * block_size + y)] = 
                            recv_buffer[x * block_size + y];
                    }
                }
                
                free(recv_buffer);
            }
        }
    } else {
        MPI_Send(local_C, block_size * block_size, MPI_INT, 0, 0, grid_comm);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    if (my_rank == 0) {
        double time = end_time - start_time;
        printf("Время исполнения: %lf\n", time * 1000);

        free(A_full);
        free(B_full);
        free(C_full);
    }
    
    free(local_A);
    free(local_B);
    free(local_C);
    MPI_Type_free(&block_type);
    MPI_Comm_free(&grid_comm);
    
    MPI_Finalize();
    return 0;
}