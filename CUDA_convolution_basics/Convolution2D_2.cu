/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

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
	int dest = threadIdx.y * blockDim.x + threadIdx.x;

	for (k = -filterR; k <= filterR; k++) {
		int d = threadIdx.x + k;

		if (d >= 0 && d < blockDim.x) {
			sum += d_Src[threadIdx.y * blockDim.x + d] * d_Filter[filterR - k];
		}     
	}

	d_Dst[dest] = sum;
}

__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter,
	int filterR) {
	
	int k;
	float sum = 0.0;
	int dest = threadIdx.y * blockDim.x + threadIdx.x;
	
	for (k = -filterR; k <= filterR; k++) {
		int d = threadIdx.y + k;

		if (d >= 0 && d < blockDim.y) {
			sum += d_Src[d * blockDim.x + threadIdx.x] * d_Filter[filterR - k];
		}   
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
	gpuErrchk(cudaMalloc( (void**) &d_Input, imageW * imageH * sizeof(float)));
	gpuErrchk(cudaMalloc( (void**) &d_Buffer, imageW * imageH * sizeof(float)));
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

	convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH,
		filter_radius);
	convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, 
		filter_radius);

	dim3 block;
	block.x = imageW;
	block.y = imageH;

	gpuErrchk(cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), 
		cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float),
		cudaMemcpyHostToDevice));

	convolutionRowGPU<<<1, block>>>(d_Buffer, d_Input, d_Filter,
		filter_radius);

	convolutionColumnGPU<<<1, block>>>(d_OutputGPU, d_Buffer, d_Filter,
		filter_radius);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW * imageH * sizeof(float), 
		cudaMemcpyDeviceToHost));

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
