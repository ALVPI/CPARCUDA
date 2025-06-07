/* Grupo G11:
	Santiago Gonzalez Rodriguez 
	Alvar Lopez Primo
*/


/*
 * This file contains both device and host code to calculate the
 * scalar product of two vectors of N elements.
 * 
 */

#include <stdio.h>

#define N 1024
#define SEGMENT_SIZE 32

///////////////////////////////////////////////////////////////////////////////
//
// Computes the scalar product of two vectors of N elements on GPU.
//
///////////////////////////////////////////////////////////////////////////////
__global__ void scalarProd(float *C, const float *A, const float *B, int nElem) {
	/*Declaration zone:*/
	/*So, for the thread id inside the grid i need:
		- thread_Idx.x = index of the thread inside his block.
		- block_idx.x = index of the block (the block of threads) inside the grid (the number of blocks that we have)
		- blockDim.x = the number of threads inside the block
		-*/
	int threadIdGrid = threadIdx.x + blockIdx.x * blockDim.x;
	/*The shared memory btw threads that we are going to use to store the partial results*/
	extern __shared__ float sharedMem[];
	/* We need to store that partial results in this variable bc we have to add each partial sum (thread_sum_result) */
	float thread_sum_result = 0.0f;
	/*Calculations*/
	if(threadIdGrid < nElem)
	{
		/*store the product of the element into the variable that is local for each thread */
		thread_sum_result = A[threadIdGrid] * B [threadIdGrid];
	}
	/*Here we have to store the sum result in the thread index position because is just the prooduct of index X in bouth matrix */
	sharedMem[threadIdx.x] = thread_sum_result;
	/*We must synchornize the threads to get sure that all the threats've finished their calculatons*/
	__syncthreads();
	/* Here we have to reduce the partials sums into the 0 position of the shared mem btw threads bc this thread will dump the results into the global mem*/

	/*Secuential mode*/
	for (int i = 1; i< blockDim.x; i++)
	{
		sharedMem[threadIdx.x] += sharedMem[i];
		__syncthreads();
	}
	/*This is the binary mode for the reduction
		// We are going to use a binary reduction, bc is the easiest way of do what we need
		//We have the "row"/2, we check that s > 0 and in each iteration we divide s/2
		for (int s = blockDim.x / 2; s > 0; s >>= 1)
		{
			//This if is the key of everything, with each itearion we are going to have less threadts
			so, we are going to have the first half of the threads as a results accumulator//
			if(threadId.x < s)
			{
				sharedMem[threadId.x] += sharedMem[threadId.x + s];
			}
			//sync the mem bc we have to be sure that each op was done
			__syncthreads();
		}
	*/
	/* The first thread of the block is in charge of write the data into de C matrix (is like the global mem for this function)*/
	if(threadIdx.x == 0)
	{
		atomicAdd(C,sharedMem[0]);
	}


}

/////////////////////////////////////////////////////////////////
//
// Computes a standard parallel reduction on GPU.
//
/////////////////////////////////////////////////////////////////
__global__ void vectorReduce(float *R, const float *C, int nElem)
{
	// Array in Shared Memory
    extern __shared__ float sdata[];
	/*For this part we need have bouth id the local in the block (threadLocalID) and the id on the grid (threadGlobalID) bc
	we have to paralelze the data and cpy from the local mem to the global mem (o surprise the C MATRIX)*/
	int threadLocalId = threadIdx.x;
	int threadGlobalId = threadIdx.x + blockIdx.x * blockDim.x;
	/*Dump the data from the C matrix into the shared memory we do this for efficency, is fastest access into the shared memory that have to acces into a matrix*/
	if(threadGlobalId< nElem)
	{
		sdata[threadLocalId] = C[threadGlobalId];
	}
	/*Sync bc we have to be sure that each thread dump the data into the shared mem*/
	__syncthreads();
	/*Secuential mode*/
	for (int i = 1; i< blockDim.x; i++)
	{
		sdata[threadLocalId] += sdata[i];
		__syncthreads();
	}
	/*This is the binary mode for the reduction
		// We are going to use a binary reduction, bc is the easiest way of do what we need
		//We have the "row"/2, we check that s > 0 and in each iteration we divide s/2
		for (int s = blockDim.x / 2; s > 0; s >>= 1)
		{
			//This if is the key of everything, with each itearion we are going to have less threadts
			so, we are going to have the first half of the threads as a results accumulator//
			if(threadLocalId < s)
			{
				sharedMem[threadLocalId] += sharedMem[threadLocalId + s];
			}
			//sync the mem bc we have to be sure that each op was done
			__syncthreads();
		}
	*/
	/* The first thread of the block is in charge of write the data into de C matrix (is like the global mem for this function)*/
	if(threadLocalId == 0)
	{
		atomicAdd(R, sdata[0]);
	}


}

// -----------------------------------------------
// Host Utility Routines
// -----------------------------------------------
float scalarProd_CPU(float *A, float *B, int nElem)
{
	float suma = 0.0f;	
	for (int i = 0; i < nElem; i++) {
		suma += A[i] * B[i];
	}
	return suma;
}

float randFloat(float low, float high) {
	float t = (float) rand() / (float) RAND_MAX;
	return (1.0f - t) * low + (t * high);
}

// ------------
// Main Program
// ------------
int main( void ) {

	// Array Elements
	int n_elem = N;
	
	// Block Dimension
	int block_dim = SEGMENT_SIZE;
	
	// Number of Blocks
	//int n_block = ( n_elem % block_dim == 0 ) // COMPLETAR...
	int n_block = (n_elem + block_dim -1) / block_dim;
	
	// Execution Configuration Parameters
	dim3 blocks (n_block, n_block);
	dim3 threads(block_dim, block_dim);
	
	// Size (in bytes) Required to Store the Matrix
	size_t n_bytes = (n_elem * sizeof(float));
	
	// Allocate Host Memory
	float *h_A = (float *) malloc(n_bytes );
	float *h_B = (float *) malloc(n_bytes );
	float *h_R = (float *) malloc(n_bytes );
		
	// Initialize Host Data
	srand(123);
	
	// Generating input data on CPU
	for (int i=0; i < n_elem; i++) {
		h_A[i] = randFloat(0.0f, 1.0f);
		h_B[i] = randFloat(0.0f, 1.0f);
	}
	
	// Compute Reference CPU Solution
	float result_cpu = scalarProd_CPU(h_A, h_B, n_elem);
	
	// CUDA Events
	cudaEvent_t start, stop;
	
	// Allocate Device Memory
	/*Here im going to explain wich is the function of this pointers on the code*/
	/*d_A the direction where the device will store the first matrix
	Same for the other ones
	d_C is used for store the partial result
	d_R is used for strore the final result*/
	float *d_A, *d_B, *d_C, *d_R;

	/*We just have to have the enougth mem to store the elements of the matrix result */
	cudaMalloc((void **)&d_A, sizeof(float)*n_elem);
	cudaMalloc((void **)&d_B, sizeof(float)*n_elem);
	cudaMalloc((void **)&d_C, sizeof(float)*n_elem);
	cudaMalloc((void **)&d_R, sizeof(float)*n_elem);
	
	// Init Events
	cudaEventCreate(&start);
	cudaEventCreate(&stop );
	
	// Start Time Measurement
    cudaEventRecord(start, 0);
	
	// Copy Host Data to Device
	/*Here i have to use cudaMemCpy->cudaMemcpy(destination, source, size, cudaMemcpyDirection);*/
	/*So here i just cpy the matrix into the graphic and we dont have to syncronyze bc cpy is a blocking op  */
	cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice);

	/*Here we invoke the function for calculate de scalarProduct
	scalarProd<<<numberOfBlocks, Block Dimension>>>(RESULT,MATRIX1, MATRIXB, numberOfElements); */
	scalarProd<<<n_block, block_dim>>>(d_C, d_A, d_B, n_elem);
	cudaDeviceSynchronize();
	/*vectorReduce<<<NumberOfBlock, block_dim, size of the shared memory)>>>(Result, source, n_element);*/
	vectorReduce<<<1, block_dim, block_dim*sizeof(float) >>>(d_R, d_C, n_elem);
	
	// Copy Device Data to Host
	/*Here i have to use cudaMemCpy->cudaMemcpy(destination, source, size, cudaMemcpyDeviceToHost);*/
	cudaMemcpy(h_R, d_R, n_bytes, cudaMemcpyDeviceToHost);
	
	
	
	// End Time Measurement
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float kernel_time;
	cudaEventElapsedTime(&kernel_time, start, stop);
    printf("Execution Time by the GPU: %.2f\n", kernel_time);

	float result_gpu = 0.0f;
	for (int i=0; i < n_block; i++) {
		result_gpu += h_R[i];
	}
	
	// Free Host Memory
	free(h_A); free(h_B); free(h_R);
	
	// Free Device Memory
	cudaFree(d_A); cudaFree(d_B);
	cudaFree(d_C); cudaFree(d_R);
	
	// Destroy Events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	if (result_cpu != result_cpu) {
		printf("Test Failed!\n");
		exit(EXIT_FAILURE);
	}
	printf("Test Passed\n");
	exit(EXIT_SUCCESS);
}
