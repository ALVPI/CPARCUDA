/* Grupo G11:
	Santiago Gonzalez Rodriguez 
	Alvar Lopez Primo
*/

/**
 * Matrix Multiplication: C = A * B.
 *
 * This file contains both device and host code to compute a matrix multiplication.
 *
 */

#include <math.h>
#include <stdio.h>

#define MATRIX_DIM   32
#define SEGMENT_SIZE 64

// --------------------
// Device Kernels
// --------------------
__global__ void transposeMatrix(float *d_data, int mat_dim) {

	// Array in Shared Memory
	extern __shared__ float sdata[];
	
	// COMPLETAR...
}

__global__ void scalarProd(float *C, const float *A, const float *B, int nElem) {

	// COMPLETAR...
}

__global__ void vectorReduce(float *R, const float *C, int nElem)
{
	// Array in Shared Memory
    extern __shared__ float sdata[];
	
	// COMPLETAR...
}

// ---------------------
// Host Utility Routines
// ---------------------
void matrixMul(const float *A, const float *B, float *C, const int n)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float acum = 0.0f;
			for (int k = 0; k < n; k++) {
				acum += A[i * n + k] * B[k * n + j];
			}
			C[i * n + j] = acum;
		}
	}
}

bool compareData(float *h_C, float *d_C, int n)
{
	double eps = 1.E-6;
	for (int i = 0; i < n * n; i++) {
		if (fabsf(h_C[i] - d_C[i]) > eps) {
			return false;
		}
	}
	return true;
}

float randFloat(float low, float high) {
	float t = (float) rand() / (float) RAND_MAX;
	return (1.0f - t) * low + (t * high);
}

// ------------
// Main Program
// ------------
int main( void ) {

	// Matrix Dimensions
	int dim_x = MATRIX_DIM; //(32 )
	int dim_y = dim_x; /*With this we get a square matrix  */
	
	// Matrix Size
	int mat_size = dim_x * dim_y;
	
	// Block Dimension
	/*Is the size of the information that a block can handle (number of threats in other words)  */
	int block_dim = SEGMENT_SIZE;//(64)
	
	// Number of Blocks
	/*Here we have to calculate the number of blocks that we need to divide the matrix
	We've to use the dimension of the matrix to know the number of blocks
	So we have, the dim_x (rows) + the number of threads the hold thing that is our worksize
	and we divide it btw the number of threats per block */
	
	int n_block = (dim_x + block_dim -1) / block_dim; 	
	/*We have to subtract 1 bc if we don't have a exact division we have to force to reserve 1 block more  to cover the hold matrix */
	
	// Execution Configuration Parameters
	/*Numeros de bloques por malla
	Number of blocks per grid (nº of blocks that we are going to use when we throw the kernel)*/
	dim3 blocksPerGrid  (block_dim, block_dim);

	/*Numero de hilos que va a tener cada bloque
	Number of threatds that a block will throw during his execution*/
	
	dim3 threadsPerBlock(n_block, n_block);
	
	// Size Required to Store the Matrix
	size_t n_bytes = (mat_size * sizeof(float));
	
	// Allocate Pinned Host Memory
	/*Pointers explanation-> every pointer is in the CPU
	-h_A the memory zone where is store the A matrix
	-h_B the memory zone where is store the B matrix
	-h_C the memory zone where is store the C matrix
	-h_R the memory zone where is store the result matrix */
	float *h_A, *h_B, *h_C, *h_R;
	/*Malloc for have the psbly of store each matrix */
	
	
	// Initialize Host Data
	srand(123);
	
	// Generating input data on CPU
	for (int i=0; i < mat_size; i++) {
		h_A[i] = randFloat(0.0f, 1.0f);
		h_B[i] = randFloat(0.0f, 1.0f);
	}
	
	// Compute Reference Matrix Multiplication
	matrixMul(h_A, h_B, h_C, dim_x);

	// CUDA Streams
	cudaStream_t stream;
	
	// Create Stream
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	
	// Performance Data
	float kernel_time, kernel_bandwidth;
	
	// Allocate Device Memory
	float *d_A, *d_B, *d_C;
	
	// COMPLETAR...	

	// CUDA Events
	cudaEvent_t start, stop;
	
	// Init Events
	cudaEventCreate(&start);
	cudaEventCreate(&stop );
	
	// Start Time Measurement
    cudaEventRecord(start, stream);
	
	// Copy Host Data to Device
	
	// COMPLETAR...	
	
	cudaStreamSynchronize(stream);
	
	transposeMatrix<<< // COMPLETAR... >>>( // COMPLETAR... );

	cudaStreamSynchronize(stream);

	for(int i = 0; i < dim_y; i++) {
		for(int j = 0; j < dim_x; j++) {
			scalarProd<<< // COMPLETAR... >>> (  // COMPLETAR... );
			cudaStreamSynchronize(stream);
			vectorReduce<<< // COMPLETAR... >>>( // COMPLETAR... );
		}
	}
	cudaDeviceSynchronize();
	
	// Copy Device Data to Host

	// COMPLETAR...	
	
	cudaStreamSynchronize(stream);
	
	// End Time Measurement
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

	float kernel_time, kernel_bandwidth;
	cudaEventElapsedTime(&kernel_time, start, stop);

	bool res = compareData(h_C, h_R, dim_x);
	
	if (res == true) {
		// Report Effective Bandwidth
		kernel_bandwidth = (2.0f * 1000.0f * n_bytes)/(1024 * 1024 * 1024);
		kernel_bandwidth /= kernel_time;
		
		printf( "Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 elements, \n",
				 kernel_bandwidth, kernel_time, (dim_x * dim_y) );
	}
	
	// Free Host Memory
	// COMPLETAR...
	
	// Free Device Memory
	// COMPLETAR...
	
	// Destroy Events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	// Destroy Stream
	cudaStreamDestroy(stream);
	
	if (res == false) {
		printf("Test Failed!\n");
		exit(EXIT_FAILURE);
	}
	printf("Test Passed\n");
	exit(EXIT_SUCCESS);
}
