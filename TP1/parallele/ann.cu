#include "ann.h"
#include "matrix.h"
#include "timers.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>

double normalRand(double mu, double sigma);
void init_weight(matrix_t *w, unsigned nneurones_prev);
void print_layer(layer_t *layer);

/**
 * @brief Donne un nombre aléatoire suivant une loi normale
 * de paramètres mu et sigma.
 * 
 * @param mu 
 * @param sigma 
 * @return double 
 */
double normalRand(double mu, double sigma)
{
    const double epsilon = DBL_MIN;
    const double two_pi = 2.0 * M_PI;
    static bool generate;
    static double z1;

    generate = !generate;

    if (!generate)
        return z1 * sigma + mu;

    double u1, u2;
    do
    {
        u1 = (double)rand() / RAND_MAX;
        u2 = (double)rand() / RAND_MAX;
    } while (u1 <= epsilon);

    double z0;
    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
}

/**
 * @brief Initialise les poids pour une couche
 * 
 * @param w - Les poids d'une couche
 * @param nneurones_prev 
 */
void init_weight(matrix_t *g_w, unsigned nneurones_prev)
{
    matrix_t *w = alloc_matrix(g_w->rows, g_w->columns);
    for (int idx = 0; idx < w->columns * w->rows; idx++)
    {
        w->m[idx] = normalRand(0, 1 / sqrt(nneurones_prev));
    }
    matrix_cudaMemcpy(g_w, w, cudaMemcpyHostToDevice);
}

/**
 * @brief Create a ann (Artifial Neural Network) object
 * 
 * @param alpha - Le pas du gradient
 * @param minibatch_size - Taille des minibatch
 * @param number_of_layers - Nombre de couches
 * @param nneurons_per_layer - Tableau du nombre de neuronnes par couches
 * @return ann_t* - L'objet ann créé
 */
ann_t *create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned *nneurons_per_layer)
{
    ann_t *nn = (ann_t *)malloc(sizeof(ann_t));

    nn->layers = (layer_t **)malloc(number_of_layers * sizeof(layer_t *));
    nn->number_of_layers = number_of_layers;
    nn->alpha = alpha;
    nn->minibatch_size = minibatch_size;

    nn->layers[0] = create_layer(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (int l = 1; l < number_of_layers; l++)
    {
        nn->layers[l] = create_layer(l, nneurons_per_layer[l], nneurons_per_layer[l - 1], minibatch_size);
    }

    return nn;
}

/**
 * @brief Create a layer object.
 * Initiliase uniquement pour les couches > 0
 * 
 * @param layer_number - Numéro de la couche
 * @param number_of_neurons - Nombre de neuronnes pour la couche
 * @param nneurons_previous_layer - Nombre de neuronnes de la couche précédente
 * @param minibatch_size - taille du minibatch
 * @return layer_t* 
 */
layer_t *create_layer(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    layer_t *layer = (layer_t *)malloc(sizeof(layer_t));

    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;
    layer->g_activations = g_alloc_matrix(number_of_neurons, minibatch_size);
    layer->g_z = g_alloc_matrix(number_of_neurons, minibatch_size);
    layer->g_delta = g_alloc_matrix(number_of_neurons, minibatch_size);
    layer->g_weights = g_alloc_matrix(number_of_neurons, nneurons_previous_layer);
    layer->g_biases = g_alloc_matrix(number_of_neurons, 1);

    if (layer_number > 0)
    {
        init_weight(layer->g_weights, nneurons_previous_layer);
    }

    return layer;
}

/**
 * @brief Copie les données d'entrée dans la 1ere couche
 * 
 * @param nn 
 * @param input 
 */
// void set_input(ann_t *nn, matrix_t *input)
// {
//     matrix_memcpy(nn->layers[0]->activations, input);
// }

/**
 * @brief Afficher les caractéristiques d'une couche
 * 
 * @param layer 
 */
void print_layer(layer_t *layer)
{
    printf("-- neurons:%d, minibatch size:%d\n", layer->number_of_neurons, layer->minibatch_size);

    printf(">> Weighted inputs --\n");
    print_g_matrix(layer->g_z, true);
    printf(">> Activations --\n");
    print_g_matrix(layer->g_activations, true);

    printf(">> Weights --\n");
    print_g_matrix(layer->g_weights, true);
    printf(">> Biases --\n");
    print_g_matrix(layer->g_biases, true);

    printf(">> Delta --\n");
    print_g_matrix(layer->g_delta, true);
}

/**
 * @brief Afficher toutes les couches d'un nn
 * 
 * @param nn 
 */
void print_nn(ann_t *nn)
{
    printf("ANN -- nlayers:%d, alpha:%lf, minibatch size: %d\n", nn->number_of_layers, nn->alpha, nn->minibatch_size);
    for (int l = 0; l < nn->number_of_layers; l++)
    {
        printf("Layer %d ", l);
        print_layer(nn->layers[l]);
    }
}

/**
 * @brief Calcul de la propgation (activations)
 * 
 * @param nn 
 * @param activation_function 
 */
void forward(ann_t *nn, matrix_t *g_one)
{
    
    for (int l = 1; l < nn->number_of_layers; l++)
    {
        // START_CUDAEVENT
        matrix_t *g_z1 = g_alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *g_z2 = g_alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        // STOP_AND_PRINT_CUDAEVENT(alloc)
        // START_CUDAEVENT
        matrix_dot(nn->layers[l]->g_weights, nn->layers[l - 1]->g_activations, g_z1); // z1 <- w^l x a^(l-1)
        // STOP_AND_PRINT_CUDAEVENT(matrix_dot1)
        // START_CUDAEVENT
        matrix_dot(nn->layers[l]->g_biases, g_one, g_z2);                             // z2 <- b^l x 1
        // STOP_AND_PRINT_CUDAEVENT(matric_dot2)
        // START_CUDAEVENT
        matrix_sum(g_z1, g_z2, nn->layers[l]->g_z);                                   // z^l <- z1 + z2 <=> z^l <- w^l x a^(l-1) + b^l x 1
        // STOP_AND_PRINT_CUDAEVENT(sum)
        // START_CUDAEVENT
        matrix_function(nn->layers[l]->g_z, false, nn->layers[l]->g_activations); // a^l = f(z^l)
        // STOP_AND_PRINT_CUDAEVENT(function)
        // START_CUDAEVENT
        g_destroy_matrix(g_z1);
        g_destroy_matrix(g_z2);
        // STOP_AND_PRINT_CUDAEVENT(destroy)
    }
}

/**
 * @brief Calcul de la rétro-propagation : mise à jour des poids et des biais
 * @todo
 * 
 * @param nn - Réseau des neurones
 * @param y - labels attendus
 * @param derivative_actfunct - fonction dérivée (dsigmoid) 
 */
void backward(ann_t *nn, matrix_t *g_y, matrix_t *g_one)
{
    unsigned L = nn->number_of_layers - 1;

    matrix_t *g_dfzL = g_alloc_matrix(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    matrix_minus(nn->layers[L]->g_activations, g_y, nn->layers[L]->g_delta);  // delta^(L) = (a^L - y)
    matrix_function(nn->layers[L]->g_z, true, g_dfzL);         // f'(z^(L))
    hadamard_product(nn->layers[L]->g_delta, g_dfzL, nn->layers[L]->g_delta); // delta^(L) = (a^L - y) o f'(z^(L))

    g_destroy_matrix(g_dfzL);

    for (int l = L; l > 1; l--)
    {
        matrix_t *g_tw, *g_delta_tmp, *g_dfz;
        g_tw = g_alloc_matrix(nn->layers[l - 1]->number_of_neurons, nn->layers[l]->number_of_neurons);
        g_delta_tmp = g_alloc_matrix(nn->layers[l - 1]->number_of_neurons, nn->minibatch_size);
        g_dfz = g_alloc_matrix(nn->layers[l - 1]->number_of_neurons, nn->minibatch_size);

        matrix_transpose(nn->layers[l]->g_weights, g_tw);                    // (w^l)T
        matrix_dot(g_tw, nn->layers[l]->g_delta, g_delta_tmp);               // (w^l)T x delta^l
        matrix_function(nn->layers[l - 1]->g_z, true, g_dfz); // f'(z^(l-1))
        hadamard_product(g_delta_tmp, g_dfz, nn->layers[l - 1]->g_delta);    // delta^(l-1) = (w^l)T x delta^l o f'(z^(l-1))

        g_destroy_matrix(g_tw);
        g_destroy_matrix(g_delta_tmp);
        g_destroy_matrix(g_dfz);
    }

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *g_w1, *g_ta;
        g_w1 = g_alloc_matrix(nn->layers[l]->number_of_neurons, nn->layers[l - 1]->number_of_neurons);
        g_ta = g_alloc_matrix(nn->minibatch_size, nn->layers[l - 1]->number_of_neurons);

        matrix_transpose(nn->layers[l - 1]->g_activations, g_ta);             // g_ta <- (a^(l-1))^T
        matrix_dot(nn->layers[l]->g_delta, g_ta, g_w1);                         // g_w1 <- deld_^l x (a^(l-1))^T
        matrix_scalar(g_w1, nn->alpha / nn->minibatch_size, g_w1);            // g_w1 <- alpha /m . delta^l x (a^(l-1))^T
        matrix_minus(nn->layers[l]->g_weights, g_w1, nn->layers[l]->g_weights); // w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T

        g_destroy_matrix(g_w1);
        g_destroy_matrix(g_ta);

        matrix_t *g_b1;
        g_b1 = g_alloc_matrix(nn->layers[l]->number_of_neurons, 1);

        matrix_dot(nn->layers[l]->g_delta, g_one, g_b1);                      // b1 <- delta^l x 1^T
        matrix_scalar(g_b1, nn->alpha / nn->minibatch_size, g_b1);            // b1 <- alpha / m . delta^l x 1^T
        matrix_minus(nn->layers[l]->g_biases, g_b1, nn->layers[l]->g_biases); // b^l = b^l - alpha / m . delta^l x 1^T

        g_destroy_matrix(g_b1);
    }
}
