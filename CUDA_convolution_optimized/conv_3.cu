/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005 
#define TILE_SIZE 32
#define ROW_TILE_SIZE 128
//#define CPU_EXEC
unsigned int row_tile_size=ROW_TILE_SIZE,col_tile_size=TILE_SIZE;

#define gpuErrchk(ans) { gpuAssert((ans), __LINE__, 1); }
inline void gpuAssert(cudaError_t code, int line, int abort) {
	if (code != cudaSuccess)  {
		fprintf(stderr,"%s, line %d\n", cudaGetErrorString(code), line);
		if (abort) {
			cudaDeviceReset();
			exit(code);
		}
	}
}

void *safeMalloc(size_t size, int line) {
    void *p = malloc(size);
    if (!p)
    {
        fprintf(stderr, "%s, line %d\n", strerror(errno), line);
        exit(EXIT_FAILURE);
    }
    return p;
}

__device__ __constant__ double d_Filter[2*512 + 1];

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
	int imageW, int imageH, int filterR) {

	int x, y, k;
											
	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			double sum = 0;

			for (k = -filterR; k <= filterR; k++) {
				int d = x + k;

				if (d >= 0 && d < imageW) {
					sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
				}     

				h_Dst[y * imageW + x] = sum;
			}
		}
	}			
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
	int imageW, int imageH, int filterR) {

	int x, y, k;
	
	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			double sum = 0;

			for (k = -filterR; k <= filterR; k++) {
				int d = y + k;

				if (d >= 0 && d < imageH) {
					sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
				}   
 
				h_Dst[y * imageW + x] = sum;
			}
		}
	}
}

__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, 
	int filterR,int row_tile_size,int unskip) {

	int k;
	double sum = 0.0;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*row_tile_size + threadIdx.x;
	int padding_skip = unskip*filterR * (2 * filterR + row_tile_size * gridDim.x) +filterR;
	int idx_x = col + padding_skip;
	int row_length = row_tile_size * gridDim.x + 2 * filterR;
	int elements_offset = row * row_length;
	int dest = idx_x + elements_offset;

	extern __shared__ double shared_data[];
	//init data
	
	shared_data[threadIdx.x] = d_Src[dest - filterR];
	__syncthreads();

	if(threadIdx.x < row_tile_size){
		for (k = -filterR; k <= filterR; k++) {
			sum += shared_data[threadIdx.x + k + filterR] * d_Filter[filterR - k];
		}
		d_Dst[dest] = sum;
	}

}

__global__ void convolutionColumnGPU(double *d_Dst, double *d_Src,
	int filterR) {
	
	int k;
	double sum = 0.0;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int row_length = blockDim.x * gridDim.x + 2 * filterR;
	int column_offset = blockIdx.x*blockDim.x + threadIdx.x + filterR;
	int dest = idx_y * blockDim.x * gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;

	extern __shared__ double shared_data[];
	//init data
	for (int i=0;i<(blockDim.y + 2*filterR);i+=blockDim.y){
		if(threadIdx.y + i < blockDim.x + 2*filterR)
			shared_data[(threadIdx.y + i)*blockDim.x + threadIdx.x] = d_Src[idx_y * row_length + column_offset +i*row_length];
	}
	__syncthreads();

	for (k = -filterR; k <= filterR; k++) {
		sum += shared_data[(threadIdx.y + filterR + k)*blockDim.x + threadIdx.x] * d_Filter[filterR - k];
	}

	d_Dst[dest] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
	double
		*h_Filter,
		*h_Input,
		*h_Buffer,
		*h_OutputGPU;

	double
		*d_Input,
		*d_Buffer,
		*d_OutputGPU;

	unsigned long int imageW;
	unsigned long int imageH;
	unsigned long int i;

	if(argc==3) {
		filter_radius = (int)strtol(argv[1], (char **)NULL, 10);
		imageW = (int)strtol(argv[2], (char **)NULL, 10);
	}
	else {
		printf("Enter filter radius : ");
		if(scanf("%d", &filter_radius) != 1) {
			printf("Scanf Failed.\n");
		}

		printf("Enter image size. Should be a power of two and greater than %d : ",
			FILTER_LENGTH);
		if(scanf("%lu", &imageW) != 1) {
			printf("Scanf Failed.\n");
		}
	}

	imageH = imageW;

	printf("Image Width x Height = %lu x %lu\n\n", imageW, imageH);
	printf("Allocating and initializing host arrays...\n");

	h_Filter    = (double *)safeMalloc(FILTER_LENGTH * sizeof(double), __LINE__);

	h_Input     = (double *)safeMalloc(imageW * imageH * sizeof(double), __LINE__);

	h_Buffer    = (double *)safeMalloc(imageW * imageH * sizeof(double), __LINE__);

	#ifdef CPU_EXEC
	double *h_OutputCPU;
	h_OutputCPU = (double *)safeMalloc(imageW * imageH * sizeof(double), __LINE__);
	#endif

	h_OutputGPU = (double *)safeMalloc(imageW * imageH * sizeof(double), __LINE__);

	srand(200);

	for (i = 0; i < FILTER_LENGTH; i++) {
		h_Filter[i] = (double)(rand() % 16);
	}

	for (i = 0; i < imageW * imageH; i++) {
		h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + 
					(double)rand() / (double)RAND_MAX;
	}

	printf("CPU computation...\n");

	#ifdef CPU_EXEC
	struct timespec start, end; 
	float time_taken; 
    clock_gettime(CLOCK_MONOTONIC, &start); 

	convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH,
		filter_radius);
	convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, 
		filter_radius);

    clock_gettime(CLOCK_MONOTONIC, &end); 
    time_taken = (end.tv_sec - start.tv_sec) * 1e9; 
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-6; 
	#endif

	dim3 grid;
	cudaDeviceProp  prop;
	cudaGetDeviceProperties(&prop,0);
	while(prop.sharedMemPerBlock <(col_tile_size+2*filter_radius)*col_tile_size*sizeof(double)){
		col_tile_size = col_tile_size/2;
	}
	while(prop.sharedMemPerBlock <(row_tile_size+2*filter_radius)*sizeof(double)){
		row_tile_size = row_tile_size/2;
	}
	int kernel_size=imageH,num_of_kernels=1;
	while(prop.totalGlobalMem <((imageW + 2 * filter_radius)* (kernel_size + 2 * filter_radius) * sizeof(double) *2 + imageW*kernel_size*sizeof(double))){
		kernel_size = kernel_size/2;
		num_of_kernels = imageH/kernel_size;
	}
	printf("Chosen kernel size %d\n",kernel_size);
	if(imageW < col_tile_size){
		grid.x = 1;
		grid.y = 1;
	}
	else{
		grid.x = imageW/col_tile_size;
		grid.y = kernel_size/col_tile_size;
	}

	if(imageW < row_tile_size){
		row_tile_size = imageW;
	}
	
	dim3 block,blockGridRows;
	block.x = imageW/grid.x;
	block.y = kernel_size/grid.y;
	blockGridRows.x = imageW/row_tile_size;
	if(num_of_kernels == 1){
		blockGridRows.y = kernel_size;
	}
	else{
		blockGridRows.y = kernel_size + filter_radius;
	}
    dim3 threadBlockRows(filter_radius + row_tile_size + filter_radius);
	dim3 blockGridRows2R(imageW/row_tile_size,kernel_size);

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	h_Buffer = (double *)realloc(h_Buffer, (imageW + 2 * filter_radius) 
		* (imageH + 2 * filter_radius) * sizeof(double));
	if (!h_Input) {
		fprintf(stderr, "%s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	gpuErrchk(cudaMalloc( (void**) &d_Input, (imageW + 2 * filter_radius)
		* (kernel_size + 2 * filter_radius) * sizeof(double)));

	gpuErrchk(cudaMalloc( (void**) &d_Buffer, (imageW + 2 * filter_radius)
		* (kernel_size + 2 * filter_radius) * sizeof(double)));

	gpuErrchk(cudaMalloc( (void**) &d_OutputGPU, imageW * kernel_size * sizeof(double)));

	memset(h_Buffer, 0, (imageW + 2 * filter_radius) 
		* (imageH + 2 * filter_radius) * sizeof(double));

	gpuErrchk(cudaMemset( (void*) d_Buffer, 0, (imageW + 2 * filter_radius)
		* (kernel_size + 2 * filter_radius) * sizeof(double)));

	for(i=0; i<imageH; i++) {
		memcpy(&h_Buffer[filter_radius * (2 * filter_radius + imageW) + 
			filter_radius + i * (2 * filter_radius + imageW)], &h_Input[i * imageW],
			imageW * sizeof(double));
	}
	int init_transfer_size = (imageW + 2 * filter_radius)*2*filter_radius +kernel_size*(imageW+2*filter_radius);
	int transfer_size = (imageW + 2* filter_radius)*kernel_size;
	int step_size = imageW*kernel_size;
	cudaEventRecord(start_event);
	gpuErrchk(cudaMemcpyToSymbol(d_Filter, h_Filter, FILTER_LENGTH*sizeof(double)));
	gpuErrchk(cudaMemcpy(d_Input, h_Buffer, init_transfer_size * sizeof(double), cudaMemcpyHostToDevice));

	convolutionRowGPU<<<blockGridRows,threadBlockRows,(row_tile_size+2*filter_radius)*sizeof(double)>>>(d_Buffer, d_Input,
		filter_radius,row_tile_size,1);
	convolutionColumnGPU<<<grid, block,col_tile_size*(col_tile_size+2*filter_radius)*sizeof(double)>>>(d_OutputGPU, d_Buffer,
		filter_radius);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW * kernel_size * sizeof(double), 
			cudaMemcpyDeviceToHost));

	for(i=1;i<num_of_kernels;i++){
		gpuErrchk(cudaMemcpy(d_Buffer,&d_Buffer[transfer_size], (init_transfer_size - transfer_size) * sizeof(double), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(&d_Input[init_transfer_size - transfer_size], &h_Buffer[init_transfer_size + (i-1)*transfer_size], transfer_size * sizeof(double), cudaMemcpyHostToDevice));
		convolutionRowGPU<<<blockGridRows2R,threadBlockRows,(row_tile_size+2*filter_radius)*sizeof(double)>>>(d_Buffer, d_Input,
			filter_radius,row_tile_size,2);
		convolutionColumnGPU<<<grid, block,col_tile_size*(col_tile_size+2*filter_radius)*sizeof(double)>>>(d_OutputGPU, d_Buffer,
		filter_radius);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaMemcpy(&h_OutputGPU[i*step_size], d_OutputGPU, imageW * kernel_size * sizeof(double), 
			cudaMemcpyDeviceToHost));
			
	}

	cudaEventRecord(stop_event);
	cudaEventSynchronize(stop_event);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start_event, stop_event);

	printf("%f\n",milliseconds);
	#ifdef CPU_EXEC
	printf("%f\n", time_taken);
	double max = ABS(h_OutputGPU[0] - h_OutputCPU[0]);
	for(i = 0; i < imageW * imageH; i++) {
		if(max < ABS(h_OutputGPU[i] - h_OutputCPU[i]))
			max = ABS(h_OutputGPU[i] - h_OutputCPU[i]);
	}
	fprintf(stderr, "Max observed error: %f\n", max);

	for(i = 0; i < imageW * imageH; i++) {
		if(ABS(h_OutputGPU[i] - h_OutputCPU[i]) > accuracy) {
			fprintf(stderr, "Images differ\n");
			break;
		}
	}
	#endif

	// free all the allocated memory

	free(h_Buffer);
	free(h_Input);
	free(h_Filter);
	free(h_OutputGPU);
	#ifdef CPU_EXEC
	free(h_OutputCPU);
	#endif

	cudaFree(d_OutputGPU);
	cudaFree(d_Buffer);
	cudaFree(d_Input);

	cudaDeviceReset();

	return 0;
}



