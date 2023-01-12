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

void test_cuda_alloc_matrix() {
    matrix_t *res = cuda_alloc_matrix(5, 3);
    print_matrix(res, false);
}

void test_matrix_dot() {
    matrix_t *A = cuda_alloc_matrix(2, 2);
    A->m[0] = 1; A->m[1] = 2; A->m[2] = 3; A->m[3] = 4;
    print_matrix(A, false);
    printf("-----------------\n");
    matrix_t *B = cuda_alloc_matrix(2, 2);
    B->m[0] = 3; B->m[1] = 6; B->m[2] = 8; B->m[3] = 0;
    print_matrix(B, false);
    printf("-----------------\n");
    matrix_t *C = cuda_alloc_matrix(2, 2);
    matrix_dot(A, B, C);
    print_matrix(C, false);
}

int main(int argc, char *argv[])
{
    // test_cuda_alloc_matrix();
    test_matrix_dot();
    return 0;
}
