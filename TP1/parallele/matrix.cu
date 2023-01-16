#include "matrix.h"
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

const int THREADS_PER_BLOCK = 16;

/**
 * @brief Allouer un espace dans la mémoire pour stocker une matrice
 * de taille rows x columns
 * 
 * @param rows 
 * @param columns 
 * @return matrix_t* 
 */
matrix_t *alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t *res = (matrix_t *)malloc(sizeof(matrix_t));
    res->m = (double *)calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
}

/**
 * @brief Allouer un espace dans la mémoire pour stocker une matrice
 * de taille rows x columns
 * 
 * @param rows 
 * @param columns 
 * @return matrix_t* 
 */
matrix_t *g_alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t *g_res = (matrix_t *)malloc(sizeof(matrix_t));
    double *m;
    cudaMalloc((double **)&m, columns * rows * sizeof(double));
    g_res->m = m;
    g_res->columns = columns;
    g_res->rows = rows;
    return g_res;
}

/**
 * @brief Libérer la mémoire pour la matrice
 * 
 * @param m 
 */
void destroy_matrix(matrix_t *m)
{
    // printf("free %p %p\n", m, m->m);
    free(m->m);
    free(m);
}

/**
 * @brief Libérer la mémoire pour la matrice
 * 
 * @param m 
 */
void g_destroy_matrix(matrix_t *m)
{
    // printf("free %p %p\n", m, m->m);
    cudaFree(m->m);
    free(m);
}

/**
 * @brief Kernel pour le produit de matrix
 * 
 * @param A 
 * @param B 
 * @param C = A * B
 * @param numRows
 * @param numColumns  
 */
__global__ void hadamard_product_kernel(double *A, double *B, double *C,int numRows, int numColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < numRows && col < numColumns) {
        C[row * numColumns + col] = A[row * numColumns + col] * B[row * numColumns + col];
    }
}

/**
 * @brief Produit de matrice (terme à terme)
 * 
 * @param m1 
 * @param m2 
 * @param res 
 */
void hadamard_product(matrix_t *g_m1, matrix_t *g_m2, matrix_t *g_res)
{
     assert((g_m1->columns == g_m2->columns) &&
            (g_m1->columns == g_res->columns) &&
            (g_m1->rows == g_m2->rows) &&
            (g_m1->rows == g_res->rows));

    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 gridDim(ceil((float)g_res->columns / blockDim.x), 
                 ceil((float)g_res->rows / blockDim.y),
                 1);

    hadamard_product_kernel<<<gridDim, blockDim>>>(g_m1->m, g_m2->m, g_res->m, g_res->rows, g_res->columns);
}

/**
 * @brief Kernel pour la somme de matrix
 * 
 * @param A 
 * @param B 
 * @param C = A + B
 * @param numRows
 * @param numColumns  
 */
__global__ void matrix_sum_kernel(double *A, double *B, double *C,int numRows, int numColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < numRows && col < numColumns) {
        C[row * numColumns + col] = A[row * numColumns + col] + B[row * numColumns + col];
    }
}

/**
 * @brief Somme de matrice (terme à terme)
 * 
 * @param m1 
 * @param m2 
 * @param res 
 */
void matrix_sum(matrix_t *g_m1, matrix_t *g_m2, matrix_t *g_res)
{
     assert((g_m1->columns == g_m2->columns) &&
            (g_m1->columns == g_res->columns) &&
            (g_m1->rows == g_m2->rows) &&
            (g_m1->rows == g_res->rows));

    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 gridDim(ceil((float)g_res->columns / blockDim.x), 
                 ceil((float)g_res->rows / blockDim.y),
                 1);

    matrix_sum_kernel<<<gridDim, blockDim>>>(g_m1->m, g_m2->m, g_res->m, g_res->rows, g_res->columns);
}

/**
 * @brief Kernel pour la différence de matrix
 * 
 * @param A 
 * @param B 
 * @param C = A + B
 * @param numRows
 * @param numColumns  
 */
__global__ void matrix_minus_kernel(double *A, double *B, double *C,int numRows, int numColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < numRows && col < numColumns) {
        C[row * numColumns + col] = A[row * numColumns + col] - B[row * numColumns + col];
    }
}

/**
 * @brief Différence de matrice (terme à terme)
 * 
 * @param m1 
 * @param m2 
 * @param res 
 */
void matrix_minus(matrix_t *g_m1, matrix_t *g_m2, matrix_t *g_res)
{
     assert((g_m1->columns == g_m2->columns) &&
            (g_m1->columns == g_res->columns) &&
            (g_m1->rows == g_m2->rows) &&
            (g_m1->rows == g_res->rows));

    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 gridDim(ceil((float)g_res->columns / blockDim.x), 
                 ceil((float)g_res->rows / blockDim.y),
                 1);

    matrix_minus_kernel<<<gridDim, blockDim>>>(g_m1->m, g_m2->m, g_res->m, g_res->rows, g_res->columns);
}

/**
 * @brief Kernel pour le produit de matrice
 * 
 * @param A 
 * @param B 
 * @param C = A * B
 * @param numARows 
 * @param numAColumns 
 * @param numBColumns 
 */
__global__ void matrix_dot_kernel(double *A, double *B, double *C, int numARows, int numAColumns, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < numARows && col < numBColumns) {
        float sum = 0;
        for (int ii = 0; ii < numAColumns; ii++) {
            sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}

/**
 * @brief Produit matriciel
 * 
 * @param m1 
 * @param m2 
 * @param res 
 */
void matrix_dot(matrix_t *g_m1, matrix_t *g_m2, matrix_t *g_res)
{
    assert((g_m1->columns == g_m2->rows) &&
           (g_m1->rows == g_res->rows) &&
           (g_m2->columns == g_res->columns));

    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 gridDim(ceil((float)g_res->columns / blockDim.x), 
                 ceil((float)g_res->rows / blockDim.y),
                 1);

    matrix_dot_kernel<<<gridDim, blockDim>>>(g_m1->m, g_m2->m, g_res->m, g_m1->rows, g_m1->columns, g_m2->columns);
}

/**
 * @brief Kernel pour l'application de fonction à une matrice
 * 
 * @param A 
 * @param B 
 * @param f 
 * @param numRows 
 * @param numColumns
 */
__global__ void matrix_function_kernel(double *A, double *B, double (*f)(double), int numRows, int numColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < numRows && col < numColumns) {
        B[row * numColumns + col] = f(A[row * numColumns + col]);
    }
}

/**
 * @brief Appliquer une fonction à tous les éléments d'une matrice.
 * 
 * @param m1 
 * @param f 
 * @param res 
 */
void matrix_function(matrix_t *g_m1, double (*f)(double), matrix_t *g_res)
{
    assert((g_m1->columns == g_res->columns) &&
           (g_m1->rows == g_res->rows));

    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 gridDim(ceil((float)g_res->columns / blockDim.x), 
                 ceil((float)g_res->rows / blockDim.y),
                 1);
    matrix_function_kernel<<<gridDim, blockDim>>>(g_m1->m, g_res->m, f, g_res->rows, g_res->columns);
}

/**
 * @brief Transposition de matrice
 * 
 * @param A 
 * @param B = At
 * @param numRows
 * @param numColumns  
 */
__global__ void matrix_transpose_kernel(double *A, double *B, int numRows, int numColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < numRows && col < numColumns) {
        B[row * numColumns + col] = A[col * numColumns + row];
    }
}

/**
 * @brief Transposition de matrice
 * 
 * @param m1 
 * @param m2 
 * @param res 
 */
void matrix_transpose(matrix_t *g_m1, matrix_t *g_res)
{
    assert((g_m1->columns == g_res->rows) &&
           (g_m1->rows == g_res->columns));

    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 gridDim(ceil((float)g_res->columns / blockDim.x), 
                 ceil((float)g_res->rows / blockDim.y),
                 1);

    matrix_transpose_kernel<<<gridDim, blockDim>>>(g_m1->m, g_res->m, g_res->rows, g_res->columns);
}

/**
 * @brief Transposition de matrice
 * 
 * @param A 
 * @param B = At
 * @param numRows
 * @param numColumns  
 */
__global__ void matrix_scalar_kernel(double *A, double s, double *res, int numRows, int numColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < numRows && col < numColumns) {
        res[row * numColumns + col] = s * A[row * numColumns + col];
    }
}

/**
 * @brief Produit de matrice par un scalaire
 * 
 * @param m1 
 * @param s 
 * @param res 
 */
void matrix_scalar(matrix_t *g_m1, double s, matrix_t *g_res)
{
    assert((g_m1->columns == g_res->columns) &&
           (g_m1->rows == g_res->rows));

    dim3 blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 gridDim(ceil((float)g_res->columns / blockDim.x), 
                 ceil((float)g_res->rows / blockDim.y),
                 1);

    matrix_scalar_kernel<<<gridDim, blockDim>>>(g_m1->m, s, g_res->m, g_res->rows, g_res->columns);
}


/**
 * @brief Copie les valaurs d'une matrice dans une autre.
 * 
 * @param dest 
 * @param src 
 */
void matrix_cudaMemcpy(matrix_t *dest, const matrix_t *src, cudaMemcpyKind kind)
{
    assert((dest->rows == src->rows) &&
           (dest->columns == src->columns));

    cudaMemcpy(dest->m, src->m, src->columns * src->rows * sizeof(double), kind);
}

/**
 * @brief Afficher la matrice
 * 
 * @param m 
 * @param is_short 
 */
void print_matrix(matrix_t *m, bool is_short)
{
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row++)
    {
        for (int col = 0; col < lim_col; col++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns)
            printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows)
        printf("...\n");
}
