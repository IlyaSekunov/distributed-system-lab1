#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MATRIX_SIZE 8  // Размер матрицы (должен быть кратен количеству процессов)
#define MAX_NUM 10

void GenerateMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() % (MAX_NUM + 1);
    }
}

void PrintMatrix(int *matrix, int rows, int cols, const char *name) {
    printf("Матрица %s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%4d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int my_rank, comm_sz;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    // Синхронизируем время начала выполнения
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    int q = (int)sqrt(comm_sz);
    int n = MATRIX_SIZE;  // Общий размер матрицы
    int block_size = n / q;  // Размер блока для каждого процесса
    
    if (my_rank == 0) {
        printf("============================================\n");
        printf("Алгоритм Кэннона для умножения матриц\n");
        printf("============================================\n");
        printf("Размер матрицы: %d x %d\n", n, n);
        printf("Количество процессов: %d (%d x %d сетка)\n", comm_sz, q, q);
        printf("Размер блока: %d x %d\n", block_size, block_size);
        printf("============================================\n\n");
    }
    
    int *local_A = calloc(block_size * block_size, sizeof(int));
    int *local_B = calloc(block_size * block_size, sizeof(int));
    int *local_C = calloc(block_size * block_size, sizeof(int));
    
    // Создаем коммуникатор для сетки процессов
    MPI_Comm grid_comm;
    int dims[2] = {q, q};
    int periods[2] = {1, 1};  // Тороидальная топология
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);
    
    // Получаем координаты текущего процесса в сетке
    int coords[2];
    MPI_Cart_coords(grid_comm, my_rank, 2, coords);
    int row = coords[0];
    int col = coords[1];
    
    // Корневой процесс инициализирует полные матрицы
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
        
        if (n <= 16) {
            PrintMatrix(A_full, n, n, "A");
            PrintMatrix(B_full, n, n, "B");
        }
    }
    
    // Создаем тип данных для блока матрицы
    MPI_Datatype block_type;
    MPI_Type_vector(block_size, block_size, n, MPI_INT, &block_type);
    MPI_Type_commit(&block_type);
    
    // Распределяем матрицы по процессам
    if (my_rank == 0) {
        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                int dest_rank;
                int dest_coords[2] = {i, j};
                MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);
                
                if (dest_rank == 0) {
                    // Копируем данные для корневого процесса
                    for (int x = 0; x < block_size; x++) {
                        for (int y = 0; y < block_size; y++) {
                            local_A[x * block_size + y] = A_full[(i * block_size + x) * n + (j * block_size + y)];
                            local_B[x * block_size + y] = B_full[(i * block_size + x) * n + (j * block_size + y)];
                        }
                    }
                } else {
                    // Отправляем блоки другим процессам
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
        // Получаем блоки от корневого процесса
        MPI_Recv(local_A, block_size * block_size, MPI_INT, 0, 0, grid_comm, MPI_STATUS_IGNORE);
        MPI_Recv(local_B, block_size * block_size, MPI_INT, 0, 1, grid_comm, MPI_STATUS_IGNORE);
    }
    
    // Начальная перестановка блоков матрицы A (сдвиг влево)
    MPI_Status status;
    int left, right, up, down;
    
    // Получаем соседей
    MPI_Cart_shift(grid_comm, 1, -1, &left, &right);  // Соседи по горизонтали
    MPI_Cart_shift(grid_comm, 0, -1, &up, &down);     // Соседи по вертикали
    
    // Сдвигаем блоки матрицы A влево на row позиций
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
    
    // Сдвигаем блоки матрицы B вверх на col позиций
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
    
    // Основной цикл алгоритма Кэннона
    for (int step = 0; step < q; step++) {
        // Локальное умножение матриц
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    local_C[i * block_size + j] += temp_A[i * block_size + k] * temp_B[k * block_size + j];
                }
            }
        }
        
        // Сдвигаем блоки матрицы A влево
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
        
        // Сдвигаем блоки матрицы B вверх
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
    
    // Собираем результаты на корневом процессе
    if (my_rank == 0) {
        // Копируем локальный результат корневого процесса
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                C_full[i * n + j] = local_C[i * block_size + j];
            }
        }
        
        // Получаем результаты от других процессов
        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                if (i == 0 && j == 0) continue;  // Корневой процесс уже обработан
                
                int source_rank;
                int source_coords[2] = {i, j};
                MPI_Cart_rank(grid_comm, source_coords, &source_rank);
                
                int *recv_buffer = (int*)malloc(block_size * block_size * sizeof(int));
                MPI_Recv(
                    recv_buffer, 
                    block_size * block_size, 
                    MPI_INT, 
                    source_rank, 
                    0, 
                    grid_comm, 
                    MPI_STATUS_IGNORE
                );
                
                // Размещаем полученный блок в результирующей матрице
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
        // Отправляем локальный результат корневому процессу
        MPI_Send(local_C, block_size * block_size, MPI_INT, 0, 0, grid_comm);
    }
    
    // Замер времени окончания выполнения
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    // Вывод результатов и информации о времени выполнения
    if (my_rank == 0) {
        if (n <= 16) {
            print_matrix(C_full, n, n, "C (результат умножения)");
        }
        
        printf("\n============================================\n");
        printf("Время выполнения: %lff мс\n", (end_time - start_time) * 1000);
        printf("============================================\n");
        
        free(A_full);
        free(B_full);
        free(C_full);
    }
    
    // Освобождаем ресурсы
    free(local_A);
    free(local_B);
    free(local_C);
    MPI_Type_free(&block_type);
    MPI_Comm_free(&grid_comm);
    
    MPI_Finalize();
    return 0;
}