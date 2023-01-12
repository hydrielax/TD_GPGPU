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


int main(int argc, char *argv[])
{
    matrix_t *res = cuda_alloc_matrix(5, 3);
    print_matrix(res, false);
    return 0;
}
