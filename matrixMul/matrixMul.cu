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
__global__ void transposeMatrix(double *aux_data, double *d_data, int mat_dim) {
      //We have to get the i and j local for the block 
        // Array in Shared Memory
        extern __shared__ double sdata[];
        /*This are the index of each cell in the GLOBAL MEMORY (d_data)*/
        int i, j, tid_block, tid_blockTras;
        j = blockIdx.x * blockDim.x + threadIdx.x;
        i = blockIdx.y * blockDim.y + threadIdx.y;
        tid_block = threadIdx.y*blockDim.x+threadIdx.x;
        tid_blockTras = threadIdx.x*blockDim.y+threadIdx.y;
        /*This are the local coords of the thread threadIdx.x & threadIdx.y
        int tid_b = j;
        int tid_g = j * mat_dim + i;*/
        //El indice local del bloque de los hilos porque la shared es el ámbito local de hilo 
        /*Now we have to dump the mem for the global memory into the shared local mem*/
        if(i < mat_dim && j < mat_dim)
        {
                /*i*mat_dim+j-> the traditional form of take the transpose position:
                bc is row*(row*col)+ column*/
                sdata[tid_block] = d_data[i*mat_dim+j];
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
                aux_data[iT * mat_dim + jT] = sdata[tid_blockTras];
        }
}

__global__ void scalarProd(double *C, const double *A, const double *B, int nElem) {
	/*Declaration zone:*/
        /*So, for the thread id inside the grid i need:
                - thread_Idx.x = index of the thread inside his block.
                - block_idx.x = index of the block (the block of threads) inside the grid (the number of blocks that we have)
                - blockDim.x = the number of threads inside the block
                -*/
        int threadIdGrid = threadIdx.x + blockIdx.x * blockDim.x;
        /*The shared memory btw threads that we are going to use to store the partial results*/
        extern __shared__ double sharedMem[];
        /* We need to store that partial results in this variable bc we have to add each partial sum (thread_sum_result) */
        double thread_sum_result = 0.0f;
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

        /*Secuential mode
        for (int i = 1; i< blockDim.x; i++)
        {
                sharedMem[threadIdx.x] += sharedMem[i];
                __syncthreads();
        }*/
        /*This is the binary mode for the reduction
                // We are going to use a binary reduction, bc is the easiest way of do what we need
                //We have the "row"/2, we check that s > 0 and in each iteration we divide s/2*/
        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
                /*This if is the key of everything, with each itearion we are going to have less threadts
                so, we are going to have the first half of the threads as a results accumulator*/
                if(threadIdx.x < s)
                {
                        sharedMem[threadIdx.x] += sharedMem[threadIdx.x + s];
                }
                /*sync the mem bc we have to be sure that each op was done*/
                __syncthreads();
        }
        /* The first thread of the block is in charge of write the data into de C matrix (is like the global mem for this function)*/
        if(threadIdx.x == 0)
        {
                atomicAdd(C,sharedMem[0]);
        }
}

__global__ void vectorReduce(double *R, const double *C, int nElem)
{
	// Array in Shared Memory
    extern __shared__ double sdata[];
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
	/*Secuential mode-> i used this code as a reference for checking if it works properly
	for (int i = 1; i< blockDim.x; i++)
	{
		sdata[threadLocalId] += sdata[i];
		__syncthreads();
	}*/
	/*This is the binary mode for the reduction
	We are going to use a binary reduction, bc is the easiest way of do what we need
	We have the "row"/2, we check that s > 0 and in each iteration we divide s/2*/
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		/*This if is the key of everything, with each itearion we are going to have less threadts
		so, we are going to have the first half of the threads as a results accumulator*/
		if(threadLocalId < s)
		{
			sdata[threadLocalId] += sdata[threadLocalId + s];
		}
		/*sync the mem bc we have to be sure that each op was done*/
		__syncthreads();
	}

	/* The first thread of the block is in charge of write the data into de C matrix (is like the global mem for this function)*/
	if(threadLocalId == 0)
	{
		atomicAdd(R, sdata[0]);
	}
}

// ---------------------
// Host Utility Routines
// ---------------------
void matrixMul(const double *A, const double *B, double *C, const int n)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double acum = 0.0f;
			for (int k = 0; k < n; k++) {
				acum += A[i * n + k] * B[k * n + j];
			}
			C[i * n + j] = acum;
		}
	}
}

bool compareData(double *h_C, double *d_C, int n)
{
	double eps = 1.E-6;
	for (int i = 0; i < n * n; i++) {
		if (fabsf(h_C[i] - d_C[i]) > eps) {
			return false;
		}
	}
	return true;
}

double randdouble(double low, double high) {
	double t = (double) rand() / (double) RAND_MAX;
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
	size_t n_bytes = (mat_size * sizeof(double));
	
	// Allocate Pinned Host Memory
	/*Pointers explanation-> every pointer is in the CPU
	-h_A the memory zone where is store the A matrix
	-h_B the memory zone where is store the B matrix
	-h_C the memory zone where is store the C matrix
	-h_R the memory zone where is store the result matrix */
	/*Malloc for have the psbly of store each matrix */
	double *h_A =(double *)malloc (n_bytes*n_bytes);
	double *h_B =(double *)malloc (n_bytes*n_bytes);
	double *h_C =(double *)malloc (n_bytes*n_bytes);
	double *h_R =(double *)malloc (n_bytes*n_bytes);
	// Initialize Host Data
	srand(123);
	
	// Generating input data on CPU
	for (int i=0; i < mat_size; i++) {
		h_A[i] = randdouble(0.0f, 1.0f);
		h_B[i] = randdouble(0.0f, 1.0f);
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
	double *d_A, *d_B, *d_C;
	double *aux_data;
	double *d_R;
	
	cudaMalloc(&d_A, n_bytes*n_bytes*sizeof(double));
	cudaMalloc(&d_B, n_bytes*n_bytes*sizeof(double));
	cudaMalloc(&d_C, n_bytes*n_bytes*sizeof(double));
	cudaMalloc(&aux_data, n_bytes*n_bytes*sizeof(double));
	cudaMalloc(&d_R, n_bytes*n_bytes*sizeof(double));



	// CUDA Events
	cudaEvent_t start, stop;
	
	// Init Events
	cudaEventCreate(&start);
	cudaEventCreate(&stop );
	
	// Start Time Measurement
    cudaEventRecord(start, stream);
	
	// Copy Host Data to Device
	cudaMemcpy(d_A, h_A, n_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, n_bytes, cudaMemcpyHostToDevice);
	
	cudaStreamSynchronize(stream);
	/*This is the invocation from transpose.cu
	transposeMatrix<<<n_block, block_dim, block_dim*sizeof(double) >>>(aux_data, d_data, dim_x);*/
	transposeMatrix<<<n_block, block_dim, block_dim*sizeof(double)>>>(aux_data, d_B, dim_x);

	cudaStreamSynchronize(stream);

	for(int i = 0; i < dim_y; i++) {
		for(int j = 0; j < dim_x; j++) {
			/*This is the invocation from scalarProd.cu
			scalarProd<<<n_block, block_dim>>>(d_C, d_A, d_B, n_elem);*/
			scalarProd<<<n_block, block_dim>>> (d_C,d_A, aux_data, mat_size);
			cudaStreamSynchronize(stream);
			/*vectorReduce<<<1, block_dim, block_dim*sizeof(double) >>>(d_R, d_C, n_elem);*/
			vectorReduce<<<1,block_dim, block_dim*sizeof(double)>>>(d_R, d_C, mat_size);
		}
	}
	cudaDeviceSynchronize();
	
	// Copy Device Data to Host

	cudaMemcpy(d_R, h_R, n_bytes, cudaMemcpyDeviceToHost);
	
	cudaStreamSynchronize(stream);
	
	// End Time Measurement
	cudaEventRecord(stop, stream);
	cudaEventSynchronize(stop);

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
	free(h_A);
	free(h_B);
	free(h_C);
	
	// Free Device Memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_R);
	cudaFree(aux_data);
	
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
