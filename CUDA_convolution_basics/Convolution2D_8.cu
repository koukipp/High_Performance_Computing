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
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
	int imageW, int imageH, int filterR) {

	int x, y, k;
											
	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			float sum = 0;

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
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
	int imageW, int imageH, int filterR) {

	int x, y, k;
	
	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			float sum = 0;

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

__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, 
	int filterR) {

	int k;
	float sum = 0.0;

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

__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter,
	int filterR) {
	
	int k;
	float sum = 0.0;

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
	float
		*h_Filter,
		*h_Input,
		*h_Buffer,
		*h_OutputCPU,
		*h_OutputGPU;

	float
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

	h_Filter    = (float *)safeMalloc(FILTER_LENGTH * sizeof(float), __LINE__);

	h_Input     = (float *)safeMalloc(imageW * imageH * sizeof(float), __LINE__);

	h_Buffer    = (float *)safeMalloc(imageW * imageH * sizeof(float), __LINE__);

	h_OutputCPU = (float *)safeMalloc(imageW * imageH * sizeof(float), __LINE__);

	h_OutputGPU = (float *)safeMalloc(imageW * imageH * sizeof(float), __LINE__);

	gpuErrchk(cudaMalloc( (void**) &d_Filter, FILTER_LENGTH * sizeof(float)));

	gpuErrchk(cudaMalloc( (void**) &d_Input, (imageW + 2 * filter_radius)
		* (imageH + 2 * filter_radius) * sizeof(float)));

	gpuErrchk(cudaMalloc( (void**) &d_Buffer, (imageW + 2 * filter_radius)
		* (imageH + 2 * filter_radius) * sizeof(float)));

	gpuErrchk(cudaMalloc( (void**) &d_OutputGPU, imageW * imageH * sizeof(float)));

	srand(200);

	for (i = 0; i < FILTER_LENGTH; i++) {
		h_Filter[i] = (float)(rand() % 16);
	}

	for (i = 0; i < imageW * imageH; i++) {
		h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + 
					(float)rand() / (float)RAND_MAX;
	}

	printf("CPU computation...\n");

    struct timespec start, end; 
	double time_taken; 
    clock_gettime(CLOCK_MONOTONIC, &start); 

	convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH,
		filter_radius);
	convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, 
		filter_radius);

    clock_gettime(CLOCK_MONOTONIC, &end); 
    time_taken = (end.tv_sec - start.tv_sec) * 1e9; 
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-6; 

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

	h_Buffer = (float *)realloc(h_Buffer, (imageW + 2 * filter_radius) 
		* (imageH + 2 * filter_radius) * sizeof(float));
	if (!h_Input) {
		fprintf(stderr, "%s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	memset(h_Buffer, 0, (imageW + 2 * filter_radius) 
		* (imageH + 2 * filter_radius) * sizeof(float));

	gpuErrchk(cudaMemset( (void*) d_Buffer, 0, (imageW + 2 * filter_radius)
		* (imageH + 2 * filter_radius) * sizeof(float)));

	for(i=0; i<imageH; i++) {
		memcpy(&h_Buffer[filter_radius * (2 * filter_radius + imageW) + 
			filter_radius + i * (2 * filter_radius + imageW)], &h_Input[i * imageW],
			imageW * sizeof(float));
	}

	cudaEventRecord(start_event);
	
	gpuErrchk(cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), 
		cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_Input, h_Buffer, (imageW + 2 * filter_radius) 
		* (imageH + 2 * filter_radius) * sizeof(float), cudaMemcpyHostToDevice));

	convolutionRowGPU<<<grid, block>>>(d_Buffer, d_Input, d_Filter,
		filter_radius);

	convolutionColumnGPU<<<grid, block>>>(d_OutputGPU, d_Buffer, d_Filter,
		filter_radius);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW * imageH * sizeof(float), 
		cudaMemcpyDeviceToHost));

	cudaEventRecord(stop_event);
	cudaEventSynchronize(stop_event);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start_event, stop_event);

	printf("%f %f\n", time_taken, milliseconds);

	float max = ABS(h_OutputGPU[0] - h_OutputCPU[0]);
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

	// free all the allocated memory

	free(h_Buffer);
	free(h_Input);
	free(h_Filter);
	free(h_OutputGPU);
	free(h_OutputCPU);

	cudaFree(d_OutputGPU);
	cudaFree(d_Buffer);
	cudaFree(d_Input);
	cudaFree(d_Filter);

	cudaDeviceReset();

	return 0;
}
