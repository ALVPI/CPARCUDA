/* Grupo G11:
	Santiago Gonzalez Rodriguez 
	Alvar Lopez Primo
*/


/*
 * Matrix Transpose
 *
 * This file contains both device and host code for transposing a matrix.
 *
 */

#include <stdio.h>
 
#define MATRIX_DIM   64
#define SEGMENT_SIZE 32

///////////////////////////////////////////////////////////
//
// Computes the Transpose of a Matrix
//
///////////////////////////////////////////////////////////
__global__ void transposeMatrix(float *d_data, int mat_dim) {

	// Array in Shared Memory
	extern __shared__ float sdata[];
	/*This are the index of each cell in the GLOBAL MEMORY (d_data)*/
	int i, j;
	j = blockIdx.x * blockDim.x + threadIdx.x;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	/*This are the local coords of the thread threadIdx.x & threadIdx.y
	int tid_b = j;
	int tid_g = j * mat_dim + i;*/
	/*Now we have to dump the mem for the global memory into the shared local mem*/
	if(i < mat_dim && j < mat_dim)
	{	
		/*i*mat_dim+j-> the traditional form of take the transpose position:
		bc is row*(row*col)+ column*/
		sdata[threadIdx.y * blockDim.x + threadIdx.x] = d_data[i*mat_dim+j];
	}
	/*We must wait untill each thread had taken their data*/
	__syncthreads();
	/*Index of the element in the transpose matrix*/
	int iT, jT;
	iT = blockIdx.x * blockDim.x + threadIdx.y;
	jT = blockIdx.y * blockDim.y + threadIdx.x;
	/*A[i][j] must b A[iT][jT] that's my goal*/
	/*So first we need to check that our position in the original matrix is valid*/
	if( iT < mat_dim && jT < mat_dim)
	{		
		/*we dumt the transpose data into the matrix (d_data)*/
		/*jT*mat_dim +iT-> the traditional form of having the transpose position*/
		d_data[jT * mat_dim + iT] = sdata[threadIdx.x * blockDim.y + threadIdx.y];
	}
}
	

// ---------------------
// Host Utility Routines
// ---------------------
void transpose(float *At, float *A, const int dim_x, const int dim_y)
{
	for (int y = 0; y < dim_y; y++) {
		for (int x = 0; x < dim_x; x++) {
			At[(x * dim_y) + y] = A[(y * dim_x) + x];
		}
	}
}

bool compareData(float *d_data, float *h_data, int n) {

	for (int i = 0; i < n * n; i++) {
		if (d_data[i] != h_data[i]) {
			return false;
		}
	}
	return true;
}

// ------------
// Main Program
// ------------
int main( void ) {

	// Matrix Dimensions
	int dim_x = MATRIX_DIM;
	int dim_y = dim_x;
	
	// Block Dimension
	int block_dim = SEGMENT_SIZE;
	
	// Number of Blocks
	int n_block = (dim_x + block_dim -1) / block_dim; 	
	// Execution Configuration Parameters
	dim3 blocksPerGrid  (n_block, n_block);
	dim3 threadsPerBlock(block_dim, block_dim);
	
	// Size (in bytes) Required to Store the Matrix
	size_t n_bytes = (dim_x * dim_y * sizeof(float));
	
	// Allocate Host Memory
	float *A = (float *) malloc(n_bytes);
	float *At = (float *) malloc(n_bytes);
	float *Aux  = (float *) malloc(n_bytes);
	
	// Initialize Host Data
	for (int i = 0; i < (dim_x * dim_y); i++) {
		A[i] = (float) i;
	}
	
	// Compute Reference Transpose Solution
	transpose(At, A, dim_x, dim_y);
	
	// CUDA Events
	cudaEvent_t start, stop;
	
	// Performance Data
	float kernel_time, kernel_bandwidth;
	
	// Allocate Device Memory
	float *d_data;
	cudaMalloc((void**)&d_data, dim_x * dim_y);

	// Init Events
	cudaEventCreate(&start);
	cudaEventCreate(&stop );
	
	// Start Time Measurement
    cudaEventRecord(start, 0);
	
	// Copy Host Data to Device
	cudaMemcpy(d_data, A, n_bytes, cudaMemcpyHostToDevice);
	
	
    transposeMatrix<<<n_block, block_dim, block_dim*sizeof(float) >>>(d_data, dim_x);
	cudaDeviceSynchronize();
	// Copy Device Data to Host
	
	cudaMemcpy(Aux, d_data, n_bytes, cudaMemcpyDeviceToHost);
    
	// End Time Measurement
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernel_time, start, stop);

	bool res = compareData(Aux, At, dim_x);
	
	if (res == true) {
		// Report Effective Bandwidth
		kernel_bandwidth = (2.0f * 1000.0f * n_bytes)/(1024 * 1024 * 1024);
		kernel_bandwidth /= kernel_time;
		
		printf( "Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 elements, \n",
				kernel_bandwidth, kernel_time, (dim_x * dim_y) );
	}
	
	// Free Host Memory
	free(A); free(At); free(Aux);
	
	// Free Device Memory
	cudaFree(d_data);
	
	// Destroy Events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	if (res == false) {
		printf("Test Failed!\n");
		exit(EXIT_FAILURE);
	}
	printf("Test Passed\n");
	exit(EXIT_SUCCESS);
}
