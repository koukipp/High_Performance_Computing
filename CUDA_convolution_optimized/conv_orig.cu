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
#define CPU_EXEC

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

__global__ void convolutionRowGPU(double *d_Dst, double *d_Src, double *d_Filter, 
	int filterR) {

	int k;
	double sum = 0.0;

	int padding_skip = filterR * (2 * filterR + blockDim.x * gridDim.x) + filterR;
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x + padding_skip;
	int row_length = blockDim.x * gridDim.x + 2 * filterR;
	int elements_offset = (blockIdx.y * blockDim.y + threadIdx.y)* row_length;
	int dest = idx_x + elements_offset;

	for (k = -filterR; k <= filterR; k++) {
		sum += d_Src[dest + k] * d_Filter[filterR - k];
	}

	d_Dst[dest] = sum;
}

__global__ void convolutionColumnGPU(double *d_Dst, double *d_Src, double *d_Filter,
	int filterR) {
	
	int k;
	double sum = 0.0;

	int padding_skip = filterR * (2 * filterR + blockDim.x * gridDim.x) + filterR;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	int row_length = blockDim.x * gridDim.x + 2 * filterR;
	int column_offset = blockIdx.x*blockDim.x + threadIdx.x + padding_skip;
	int dest = idx_y * blockDim.x * gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;

	for (k = -filterR; k <= filterR; k++) {
		sum += d_Src[(idx_y + k) * row_length + column_offset] * d_Filter[filterR - k];
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
		*d_Filter,
		*d_Input,
		*d_Buffer,
		*d_OutputGPU;

	int imageW;
	int imageH;
	unsigned int i;

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
		if(scanf("%d", &imageW) != 1) {
			printf("Scanf Failed.\n");
		}
	}

	imageH = imageW;

	printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
	printf("Allocating and initializing host arrays...\n");

	h_Filter    = (double *)safeMalloc(FILTER_LENGTH * sizeof(double), __LINE__);

	h_Input     = (double *)safeMalloc(imageW * imageH * sizeof(double), __LINE__);

	h_Buffer    = (double *)safeMalloc(imageW * imageH * sizeof(double), __LINE__);

	#ifdef CPU_EXEC
	double *h_OutputCPU;
	h_OutputCPU = (double *)safeMalloc(imageW * imageH * sizeof(double), __LINE__);
	#endif

	h_OutputGPU = (double *)safeMalloc(imageW * imageH * sizeof(double), __LINE__);

	gpuErrchk(cudaMalloc( (void**) &d_Filter, FILTER_LENGTH * sizeof(double)));

	gpuErrchk(cudaMalloc( (void**) &d_Input, (imageW + 2 * filter_radius)
		* (imageH + 2 * filter_radius) * sizeof(double)));

	gpuErrchk(cudaMalloc( (void**) &d_Buffer, (imageW + 2 * filter_radius)
		* (imageH + 2 * filter_radius) * sizeof(double)));

	gpuErrchk(cudaMalloc( (void**) &d_OutputGPU, imageW * imageH * sizeof(double)));

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
	if(imageW < 32){
		grid.x = 1;
		grid.y = 1;
	}
	else{
		grid.x = imageW/32;
		grid.y = imageH/32;
	}
	
	dim3 block;
	block.x = imageW/grid.x;
	block.y = imageH/grid.y;

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	h_Buffer = (double *)realloc(h_Buffer, (imageW + 2 * filter_radius) 
		* (imageH + 2 * filter_radius) * sizeof(double));
	if (!h_Input) {
		fprintf(stderr, "%s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	memset(h_Buffer, 0, (imageW + 2 * filter_radius) 
		* (imageH + 2 * filter_radius) * sizeof(double));

	gpuErrchk(cudaMemset( (void*) d_Buffer, 0, (imageW + 2 * filter_radius)
		* (imageH + 2 * filter_radius) * sizeof(double)));

	for(i=0; i<imageH; i++) {
		memcpy(&h_Buffer[filter_radius * (2 * filter_radius + imageW) + 
			filter_radius + i * (2 * filter_radius + imageW)], &h_Input[i * imageW],
			imageW * sizeof(double));
	}

	cudaEventRecord(start_event);
	
	gpuErrchk(cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(double), 
		cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_Input, h_Buffer, (imageW + 2 * filter_radius) 
		* (imageH + 2 * filter_radius) * sizeof(double), cudaMemcpyHostToDevice));

	convolutionRowGPU<<<grid, block>>>(d_Buffer, d_Input, d_Filter,
		filter_radius);

	convolutionColumnGPU<<<grid, block>>>(d_OutputGPU, d_Buffer, d_Filter,
		filter_radius);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW * imageH * sizeof(double), 
		cudaMemcpyDeviceToHost));

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
	cudaFree(d_Filter);

	cudaDeviceReset();

	return 0;
}
