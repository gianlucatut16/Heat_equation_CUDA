#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#define N 10000                   // Size of the grid
#define numSteps 1000           // Number of iteration
#define ALPHA 0.1               // Heat equation constant
#define SQUARE_SIZE 1000          // Dimension of the initial heated square
#define BLOCK_SIZE 32              // Thread block size for kernel
#define OutputNum 200           // Number of iterations to print grid

__global__ void heatEquation(float* u_old, float* u_new, const float dx2, const float dy2, const float dt) {
    // Thread indexes in the grid
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < N - 1 && j < N - 1) {
        int idx = j * N + i;     // Linear index

        // Heat equation
        u_new[idx] = u_old[idx] + ALPHA * dt * ( (u_old[idx - 1] - 2.0 * u_old[idx] + u_old[idx + 1])/dx2 + 
                                                 (u_old[idx - N] - 2.0 * u_old[idx] + u_old[idx + N])/dy2 );
    }
}

void GridToFile(const char* filename, float* grid) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Failed to open file '%s' for writing.\n", filename);
        return;
    }

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {

            fprintf(file, "%f ", grid[j * N + i]);
        }

        fprintf(file, "\n");
    }

    fclose(file);
}

int main() {

    // Deltas
    const float dx = 0.01;   // Horizontal grid spacing 
    const float dy = 0.01;   // Vertical grid spacing

    const float dx2 = dx*dx;
    const float dy2 = dy*dy;

    const float dt = dx2 * dy2 / (2.0 * ALPHA * (dx2 + dy2)); // Largest stable time step

    // Declaring the inital and final grids
    float* u_old, *u_new;
    float* d_u_old, *d_u_new;

    u_old = (float*)malloc(N * N * sizeof(float));
    u_new = (float*)malloc(N * N * sizeof(float));

    // Initialize the grid
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            int idx = j * N + i;
            if (i >= (N - SQUARE_SIZE) / 2 && i < (N + SQUARE_SIZE) / 2 &&
                j >= (N - SQUARE_SIZE) / 2 && j < (N + SQUARE_SIZE) / 2) {
                u_old[idx] = 1.0;
            } else {
                u_old[idx] = 0.0;
            }
        }
    }

    // // Writing the initial grid to a file
    // GridToFile("heat_0000.txt", u_old);

    // Allocating memory on the GPU
    cudaMalloc((void**)&d_u_old, N * N * sizeof(float));
    cudaMalloc((void**)&d_u_new, N * N * sizeof(float));

    // Copying the initial grid to the GPU
    cudaMemcpy(d_u_old, u_old, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Threaed block size and thread block's grid size
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N-1) / (dimBlock.x + 1), (N-1) / (dimBlock.y + 1));

    // Declaring the variables to measure execssution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Starting the timer
    cudaEventRecord(start);

    // Launching the kernel
    for (int t = 0; t <= numSteps; t++) {
        heatEquation<<<dimGrid, dimBlock>>>(d_u_old, d_u_new, dx2, dy2, dt);

        // Swap the old and new grids
        float* temp = d_u_new;
        d_u_new = d_u_old;
        d_u_old = temp;

        // // Output visualization
        // if (t % OutputNum == 0 && t != 0){
        //     // Copying the final grid back to the CPU
        //     cudaMemcpy(u_new, d_u_old, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        //     // Writing the final grid to a file
        //     char filename[64];
        //     sprintf(filename, "heat_%04d.txt", t);
        //     GridToFile(filename, u_new);
        // }
    }

    // Stopping the timer
    cudaEventRecord(stop);

    // Synchronizing threads
    cudaEventSynchronize(stop);

    // Saving the execution time
    float exec_time = 0;
    cudaEventElapsedTime(&exec_time, start, stop);
    exec_time /= 1000;  //Execution time in seconds

    // Copying the final grid back to the CPU
    cudaMemcpy(u_new, d_u_old, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Showing the parallel execution time
    printf("Parallel execution time: %f s \n", exec_time);

    // Free memory
    free(u_old);
    free(u_new);
    cudaFree(d_u_old);
    cudaFree(d_u_new);

    return 0;
}

