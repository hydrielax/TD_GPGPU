#include "matrix.h"
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

const int THREADS_PER_BLOCK = 256;

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
matrix_t *cuda_alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t *res;
    cudaMallocManaged(&res, sizeof(matrix_t));
    double *m;
    cudaMallocManaged(&m, columns * rows * sizeof(double));
    res->m = m;
    res->columns = columns;
    res->rows = rows;
    for (int idx = 0; idx < columns * rows; idx++)
    {
        res->m[idx] = 0;
    }
    return res;
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

/**
 * @brief Calculer le produit de Hadamard (produit terme à terme)
 * 
 * @param m1 
 * @param m2 
 * @param res 
 */
void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->columns) &&
           (m1->columns == res->columns) &&
           (m1->rows == m2->rows) &&
           (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx++)
    {
        res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

/**
 * @brief Somme de matrice (terme à terme)
 * 
 * @param m1 
 * @param m2 
 * @param res 
 */
void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->columns) &&
           (m1->columns == res->columns) &&
           (m1->rows == m2->rows) &&
           (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx++)
    {
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

/**
 * @brief Différence de matrices: m1 - m2
 * 
 * @param m1 
 * @param m2 
 * @param res 
 */
void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->columns) &&
           (m1->columns == res->columns) &&
           (m1->rows == m2->rows) &&
           (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
}


__global__ void computeMatrixMulGPU(double *A, double *B, double *C, int d1, int d2, int d3)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x; // Col
    int i = threadIdx.y + blockIdx.y * blockDim.y; // Row
    if (i < d1 && j < d3)
    {
        double s = 0;
        for (int k = 0; k < d2; k++)
        {
            s += A[i * d2 + k] * B[j + k * d3];
        }
        C[i * d3 + j] = s;
    }
}


/**
 * @brief Produit matriciel
 * 
 * @param m1 
 * @param m2 
 * @param res 
 */
void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->rows) &&
           (m1->rows == res->rows) &&
           (m2->columns == res->columns));

    int gridSize = ceil((float)res->rows * res->columns / THREADS_PER_BLOCK);
    dim3 DimGrid(gridSize, gridSize, 1);
    dim3 DimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    computeMatrixMulGPU<<<DimGrid, DimBlock>>>(m1->m, m2->m, res->m, m1->rows, m1->columns, m2->columns);
    cudaDeviceSynchronize();
    // for (int row = 0; row < m1->rows; row++)
    // {
    //     for (int col = 0; col < m2->columns; col++)
    //     {
    //         int idx = col + row * m2->columns;
    //         double var = 0.0;

    //         for (int ii = 0; ii < m1->columns; ii++)
    //         {
    //             var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
    //         }

    //         res->m[idx] = var;
    //     }
    // }
}

/**
 * @brief Appliquer une fonction à tous les éléments d'une matrice.
 * 
 * @param m1 
 * @param f 
 * @param res 
 */
void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert((m1->columns == res->columns) &&
           (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}


/**
 * @brief Transposer une matrice
 * 
 * @param m1 
 * @param res 
 */
void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert((m1->columns == res->rows) &&
           (m1->rows == res->columns));

    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

/**
 * @brief Produit externe par un scalaire
 * 
 * @param m1 
 * @param s 
 * @param res 
 */
void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert((m1->rows == res->rows) &&
           (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns * m1->rows; idx++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

/**
 * @brief Copie les valaurs d'une matrice dans une autre.
 * 
 * @param dest 
 * @param src 
 */
void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert((dest->rows == src->rows) &&
           (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));
}