#include <iostream>
#include <math.h>
#include <assert.h>

using namespace std;

void initMatrix(float *m, int numRows, int numCols);
void computeMatrixMulCPU(float *A, float *B, float *C, int d1, int d2, int d3);
void compareMatrix(float *A, float *B, int numRows, int numColumns);

__global__ void computeMatrixMulGPU(float *A, float *B, float *C, int d1, int d2, int d3)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x; // Col
    int i = threadIdx.y + blockIdx.y * blockDim.y; // Row
    if (i < d1 && j < d3)
    {
        float s = 0;
        for (int k = 0; k < d2; k++)
        {
            s += A[i * d2 + k] * B[j + k * d3];
        }
        C[i * d3 + j] = s;
    }
}

int main(int argc, char *argv[])
{
    int numARows = atoi(argv[1]);    // number of rows in the matrix A
    int numAColumns = atoi(argv[2]); // number of columns in the matrix A
    int numBRows = atoi(argv[3]);    // number of rows in the matrix B
    int numBColumns = atoi(argv[4]); // number of columns in the matrix B
    int numCRows = numARows;         // number of rows in the matrix C
    int numCColumns = numBColumns;   // number of columns in the matrix C
    assert(numAColumns == numBRows);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float *A = (float *)malloc(numARows * numAColumns * sizeof(float));
    float *B = (float *)malloc(numBRows * numBColumns * sizeof(float));
    float *C = (float *)malloc(numCRows * numCColumns * sizeof(float));
    float *D = (float *)malloc(numCRows * numCColumns * sizeof(float));

    // Initialize matrices on the host
    initMatrix(A, numARows, numAColumns);
    initMatrix(B, numBRows, numBColumns);

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc((void **)&d_A, numARows * numAColumns * sizeof(float));
    cudaMalloc((void **)&d_B, numBRows * numBColumns * sizeof(float));
    cudaMalloc((void **)&d_C, numCRows * numCColumns * sizeof(float));

    cudaMemcpy(d_A, A, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, numCRows * numCColumns * sizeof(float), cudaMemcpyHostToDevice);

    dim3 DimGrid(ceil((float)numCRows * numCColumns / 16.0), ceil((float)numCRows * numCColumns / 16.0), 1);
    dim3 DimBlock(16, 16, 1);

    computeMatrixMulGPU<<<DimGrid, DimBlock>>>(A, B, C, numARows, numAColumns, numBColumns);
    cudaMemcpy(C, d_C, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // computeMatrixMulCPU(A, B, D, numARows, numAColumns ,numBColumns);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to compute %3.1f ms\n", elapsedTime);

    // compareMatrix(C, D, numCRows, numCColumns);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}

void initMatrix(float *m, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            m[i * numCols + j] = sin(i * numCols + j);
        }
    }
}

void computeMatrixMulCPU(float *A, float *B, float *C, int d1, int d2, int d3)
{
    for (int i = 0; i < d1; i++)
    {
        for (int j = 0; j < d3; j++)
        {
            int s = 0;
            for (int k = 0; k < d2; k++)
            {
                s += A[i * d2 + k] * B[j + k * d3];
            }
            C[i * d3 + j] = s;
        }
    }
}

void compareMatrix(float *A, float *B, int numRows, int numColumns)
{
    for (int row = 0; row < numRows; row++)
    {
        for (int col = 0; col < numColumns; col++)
        {
            assert(A[row * numColumns + col] == B[row * numColumns + col]);
        }
    }
    cout << "The matrices are identical" << endl;
}