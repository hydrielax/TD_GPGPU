#ifndef ANN_H
#define ANN_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "matrix.h"

typedef struct
{
    unsigned minibatch_size;
    unsigned number_of_neurons;

    matrix_t *g_weights;
    matrix_t *g_biases;

    matrix_t *g_z;
    matrix_t *g_activations;

    matrix_t *g_delta;
} layer_t;

typedef struct
{
    void (*f)(double *, double *, unsigned, unsigned);
    void (*fd)(double *, double *, unsigned, unsigned);
    double alpha;
    unsigned minibatch_size;
    unsigned input_size;
    unsigned number_of_layers;
    layer_t **layers;
} ann_t;

ann_t *create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned *nneurons_per_layer);

layer_t *create_layer(unsigned l, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size);

// void set_input(ann_t *nn, matrix_t *input);

void print_nn(ann_t *nn);

void forward(ann_t *nn, matrix_t *g_one);

void backward(ann_t *nn, matrix_t *y, matrix_t *g_one);

#endif