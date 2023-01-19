#ifndef MATRIX_H
#define MATRIX_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

typedef struct
{
    double *m;
    unsigned columns;
    unsigned rows;
} matrix_t;

matrix_t *alloc_matrix(unsigned rows, unsigned columns);
matrix_t *g_alloc_matrix(unsigned rows, unsigned columns);

void destroy_matrix(matrix_t *m);
void g_destroy_matrix(matrix_t *m);

void print_matrix(matrix_t *m, bool is_short);
void print_g_matrix(matrix_t *g_m, bool is_short);

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_function(matrix_t *m1, bool derivee, matrix_t *res);

void matrix_transpose(matrix_t *m1, matrix_t *res);

void matrix_scalar(matrix_t *m1, double s, matrix_t *res);

void matrix_cudaMemcpy(matrix_t *dest, const matrix_t *src, cudaMemcpyKind kind);

#endif