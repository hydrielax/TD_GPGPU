// Compile nvcc -o ./test test.cu matrix.cu ann.cu mnist.cu -lm

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "matrix.h"
#include "ann.h"
#include "timers.h"
#include <math.h>
#include <string.h>
#include <time.h>

void test_alloc_matrix() {
    matrix_t *res = alloc_matrix(5, 3);
    print_matrix(res, false);
}

void test_matrix_dot() {
    // matrice A
    matrix_t *A = alloc_matrix(3, 2);
    matrix_t *g_A = g_alloc_matrix(3, 2);
    A->m[0] = 1; A->m[1] = 2; A->m[2] = 3; A->m[3] = 4; A->m[4] = 5; A->m[5] = 6;
    matrix_cudaMemcpy(g_A, A, cudaMemcpyHostToDevice);
    print_matrix(A, false);
    printf("-----------------\n");
    // matrice B
    matrix_t *B = alloc_matrix(2, 2);
    matrix_t *g_B = g_alloc_matrix(2, 2);
    B->m[0] = 3; B->m[1] = 6; B->m[2] = 8; B->m[3] = 0;
    matrix_cudaMemcpy(g_B, B, cudaMemcpyHostToDevice);
    print_matrix(B, false);
    printf("-----------------\n");
    // matrice C
    matrix_t *C = alloc_matrix(3, 2);
    matrix_t *g_C = g_alloc_matrix(3, 2);
    matrix_dot(g_A, g_B, g_C);
    matrix_cudaMemcpy(C, g_C, cudaMemcpyDeviceToHost);
    print_matrix(C, false);
}


void test_matrix_scalar() {
    // matrice A
    matrix_t *A = alloc_matrix(3, 2);
    matrix_t *g_A = g_alloc_matrix(3, 2);
    A->m[0] = 1; A->m[1] = 2; A->m[2] = 3; A->m[3] = 4; A->m[4] = 5; A->m[5] = 6;
    matrix_cudaMemcpy(g_A, A, cudaMemcpyHostToDevice);
    print_matrix(A, false);
    printf("-----------------\n");
    // scalaire
    double scalar = 1.5;
    // matrice C
    matrix_t *C = alloc_matrix(3, 2);
    matrix_t *g_C = g_alloc_matrix(3, 2);
    matrix_scalar(g_A, scalar, g_C);
    matrix_cudaMemcpy(C, g_C, cudaMemcpyDeviceToHost);
    print_matrix(C, false);
}


void test_matrix_transpose() {
    // matrice A
    matrix_t *A = alloc_matrix(3, 2);
    matrix_t *g_A = g_alloc_matrix(3, 2);
    A->m[0] = 1; A->m[1] = 2; A->m[2] = 3; A->m[3] = 4; A->m[4] = 5; A->m[5] = 6;
    matrix_cudaMemcpy(g_A, A, cudaMemcpyHostToDevice);
    print_matrix(A, false);
    printf("-----------------\n");
    // matrice C
    matrix_t *C = alloc_matrix(2, 3);
    matrix_t *g_C = g_alloc_matrix(2, 3);
    matrix_transpose(g_A, g_C);
    matrix_cudaMemcpy(C, g_C, cudaMemcpyDeviceToHost);
    print_matrix(C, false);
}


void test_matrix_minus() {
    // matrice A
    matrix_t *A = alloc_matrix(2, 2);
    matrix_t *g_A = g_alloc_matrix(2, 2);
    A->m[0] = 1; A->m[1] = 2; A->m[2] = 3; A->m[3] = 4;
    matrix_cudaMemcpy(g_A, A, cudaMemcpyHostToDevice);
    print_matrix(A, false);
    printf("-----------------\n");
    // matrice B
    matrix_t *B = alloc_matrix(2, 2);
    matrix_t *g_B = g_alloc_matrix(2, 2);
    B->m[0] = 3; B->m[1] = 6; B->m[2] = 8; B->m[3] = 0;
    matrix_cudaMemcpy(g_B, B, cudaMemcpyHostToDevice);
    print_matrix(B, false);
    printf("-----------------\n");
    // matrice C
    matrix_t *C = alloc_matrix(2, 2);
    matrix_t *g_C = g_alloc_matrix(2, 2);
    matrix_minus(g_A, g_B, g_C);
    matrix_cudaMemcpy(C, g_C, cudaMemcpyDeviceToHost);
    print_matrix(C, false);
}


int main(int argc, char *argv[])
{
    // test_alloc_matrix();
    test_matrix_transpose();
    return 0;
}
