#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>

__global__ void vectorAdd(float *a, float *b, float *c, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
    {
        printf("Index: %d\n", index);
        c[index] = a[index] + b[index];
    }
}

int main()
{
    int n = 1000; // Size of vectors
    size_t bytes = n * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // Initialize vectors
    for (int i = 0; i < n; i++)
    {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Define kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < n; i++)
    {
        if (h_c[i] != h_a[i] + h_b[i])
        {
            std::cerr << "Error: Incorrect result at index " << i << std::endl;
            break;
        }
        std::cout << "h_c[" << i << "] = " << h_c[i] << std::endl;
    }

    std::cout << "Vector addition successful!" << std::endl;

    // Free device and host memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
