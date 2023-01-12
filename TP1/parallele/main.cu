// Compile nvcc -o ./ann main.cu matrix.cu ann.cu mnist.cu -lm

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


void populate_minibatch(double *x, double *y, unsigned *minibatch_idx, unsigned minibatch_size, image *img, unsigned img_size, byte *label, unsigned label_size);

/**
 * @brief Créer le tableau [0:n]
 * 
 * @param n - valeur max
 * @param t - adresse du tableau
 */
void zero_to_n(unsigned n, unsigned *t)
{
    for (unsigned i = 0; i < n; i++)
    {
        t[i] = i;
    }
}

/**
 * @brief Effectue number_of_switch permutations sur un tableau
 * 
 * @param t - le tableau
 * @param size 
 * @param number_of_switch - nombre de permutations
 */
void shuffle(unsigned *t, const unsigned size, const unsigned number_of_switch)
{
    zero_to_n(size, t);
    for (unsigned i = 0; i < number_of_switch; i++)
    {
        unsigned x = rand() % size;
        unsigned y = rand() % size;
        unsigned tmp = t[x];
        t[x] = t[y];
        t[y] = tmp;
    }
}

/**
 * @brief Fonction sigmoïde
 * 
 * @param x 
 * @return double 
 */
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

/**
 * @brief Dérivée de la fonction d'activation sigmoïde
 * 
 * @param x 
 * @return double 
 */
double dsigmoid(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

/**
 * @brief Calcule l'accuracy du réseau de neurones sur le jeu de test
 * 
 * @param test_img 
 * @param test_label 
 * @param datasize 
 * @param minibatch_size 
 * @param nn 
 * @return double - pourcentage de réussites
 */
double accuracy(image *test_img, byte *test_label, unsigned datasize, unsigned minibatch_size, ann_t *nn)
{
    unsigned good = 0;
    unsigned idx[datasize];
    double *x = (double *)malloc(28 * 28 * minibatch_size * sizeof(double));
    double *y = (double *)malloc(10 * minibatch_size * sizeof(double));

    zero_to_n(datasize, idx);

    for (int i = 0; i < datasize - minibatch_size; i += minibatch_size)
    {
        populate_minibatch(x, y, &idx[i], minibatch_size, test_img, 28 * 28, test_label, 10);
        memcpy(nn->layers[0]->activations->m, x, 28 * 28 * minibatch_size * sizeof(double));

        forward(nn, sigmoid);
        for (int col = 0; col < minibatch_size; col++)
        {
            int idxTrainingData = col + i;
            double max = 0;
            unsigned idx_max = 0;
            for (int row = 0; row < 10; row++)
            {
                int idx = col + row * minibatch_size;
                if (nn->layers[nn->number_of_layers - 1]->activations->m[idx] > max)
                {
                    max = nn->layers[nn->number_of_layers - 1]->activations->m[idx];
                    idx_max = row;
                }
            }
            if (idx_max == test_label[idxTrainingData])
            {
                good++;
            }
        }
    }
    free(x);
    free(y);

    unsigned ntests = (datasize / minibatch_size) * minibatch_size;
    return (100.0 * (double)(good) / ntests);
}

/**
 * @brief Crée un minibatch (X, Y)
 * @todo
 * 
 * @param x 
 * @param y 
 * @param minibatch_idx - liste des id des images pour ce minibatch
 * @param minibatch_size - taille du minibatch (nombre d'images)
 * @param img - liste de toutes les images
 * @param img_size - taille d'une image
 * @param label - liste de tous les labels
 * @param label_size - nombre de labels (classes) possibles
 */
void populate_minibatch(double *x, double *y, unsigned *minibatch_idx, unsigned minibatch_size, image *img, unsigned img_size, byte *label, unsigned label_size)
{
    for (int col = 0; col < minibatch_size; col++)
    {
        for (int row = 0; row < img_size; row++)
        {
            x[row * minibatch_size + col] = (double)img[minibatch_idx[col]][row] / 255.;
        }

        for (int row = 0; row < label_size; row++)
        {
            y[row * minibatch_size + col] = 0.0;
        }

        y[label[minibatch_idx[col]] * minibatch_size + col] = 1.0;
    }
}

/**
 * @brief Fonction qui fait tout
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char *argv[])
{
    srand(time(0));
    unsigned datasize, ntest;
    image *train_img = read_images("train-images-idx3-ubyte", &datasize);
    byte *train_label = read_labels("train-labels-idx1-ubyte", &datasize);
    image *test_img = read_images("t10k-images-idx3-ubyte", &ntest);
    byte *test_label = read_labels("t10k-labels-idx1-ubyte", &ntest);

    ann_t *nn;
    double alpha = 0.05;
    unsigned minibatch_size = 16;
    unsigned number_of_layers = 3;
    unsigned nneurons_per_layer[3] = {28 * 28, 30, 10};
    nn = create_ann(alpha, minibatch_size, number_of_layers, nneurons_per_layer);
    // print_nn(nn);

    printf("starting accuracy %lf\n", accuracy(test_img, test_label, ntest, minibatch_size, nn));

    unsigned *shuffled_idx = (unsigned *)malloc(datasize * sizeof(unsigned));
    double *x = (double *)malloc(28 * 28 * minibatch_size * sizeof(double));
    double *y = (double *)malloc(10 * minibatch_size * sizeof(double));
    matrix_t *out = alloc_matrix(10, minibatch_size);

    CREATE_CUDAEVENT

    for (int epoch = 0; epoch < 40; epoch++)
    {
        printf("start learning epoch %d\n", epoch);

        shuffle(shuffled_idx, datasize, datasize);

        for (int i = 0; i < datasize - minibatch_size; i += minibatch_size)
        {
            populate_minibatch(x, y, shuffled_idx + i, minibatch_size, train_img, 28 * 28, train_label, 10);
            memcpy(nn->layers[0]->activations->m, x, 28 * 28 * minibatch_size * sizeof(double));
            forward(nn, sigmoid);
            memcpy(out->m, y, 10 * minibatch_size * sizeof(double));
            backward(nn, out, dsigmoid);
        }

        double acc = accuracy(test_img, test_label, ntest, minibatch_size, nn);
        printf("\tepoch %d accuracy %lf\n", epoch, acc);
    }

    free(x);
    free(y);
    free(shuffled_idx);
    destroy_matrix(out);

    return 0;
}
