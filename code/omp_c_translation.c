/*
 * File: omp_c_translation.c
 * Description: A C-based translation of the MATLAB OMP algorithm.
 * Demonstrates the core logic of finding the maximum correlation 
 * and updating the residual vector for embedded deployment.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define M 4 // Number of measurements
#define N 8 // Length of the sparse vector (Dictionary size)
#define S 2 // Sparsity level

// Helper function to find the index of the maximum value in an array
int argmax(float arr[], int size) {
    int max_idx = 0;
    float max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int main() {
    // 1. Mock Dictionary Matrix (A) and Measurement Vector (y)
    // In a real system, A would be your DFT/Pilot matrix, y would be received signal
    float A[M][N] = {
        {0.1, 0.9, 0.2, 0.1, 0.3, 0.1, 0.1, 0.1},
        {0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1},
        {0.1, 0.1, 0.2, 0.9, 0.1, 0.1, 0.1, 0.1},
        {0.1, 0.1, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1}
    };
    
    float y[M] = {1.0, 0.9, 1.1, 0.95}; // Received signal (mostly composed of columns 1 and 3)
    
    float r[M]; // Residual
    for(int i=0; i<M; i++) r[i] = y[i]; 
    
    int active_set[S] = {-1, -1}; // To store chosen column indices
    float correlations[N] = {0};

    printf("Starting OMP Algorithm (Sparsity S=%d)\n", S);

    // 2. OMP Iteration Loop
    for (int iter = 0; iter < S; iter++) {
        
        // A. Compute inner product (correlation) of residual with every column in A
        for (int j = 0; j < N; j++) {
            float dot_product = 0;
            for (int i = 0; i < M; i++) {
                dot_product += A[i][j] * r[i];
            }
            correlations[j] = fabs(dot_product); // Absolute value for magnitude
        }
        
        // B. Find the column with the highest correlation
        int best_idx = argmax(correlations, N);
        active_set[iter] = best_idx;
        printf("Iteration %d: Selected dictionary index %d\n", iter+1, best_idx);
        
        // C. (Simplified) Update Residual: Subtract the projection
        // In a real implementation, this requires solving a Least Squares problem 
        // using Gram-Schmidt or QR decomposition (e.g., using CMSIS-DSP library).
        // For this C prototype, we use a simple gradient descent step to approximate.
        float projection = 0;
        for (int i = 0; i < M; i++) {
            projection += A[i][best_idx] * r[i];
        }
        
        for (int i = 0; i < M; i++) {
            r[i] = r[i] - (projection * A[i][best_idx]);
        }
        
        // Suppress the found index so it's not picked again
        correlations[best_idx] = -1.0; 
    }

    printf("OMP Complete. Active support set: [%d, %d]\n", active_set[0], active_set[1]);
    
    return 0;
}
