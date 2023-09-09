#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

#define N 256

__global__ void histogram_gpu(int *hist_out, unsigned char *img_in){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	atomicAdd(&hist_out[img_in[i]], 1);
}

__global__ void prefix_sum(int **hists_in, int *lut, int img_size) {
	int pin = 1, pout = 0, j, i = threadIdx.x;
	__shared__ int buffer[2*N], min;

	buffer[i] =  hists_in[0][i] + hists_in[1][i];
	
	__syncthreads();

	if(i==0) {
		j=0;
		min=0;
		while(min == 0){
			min = buffer[j++];
		}
	}

	for(j=1; j<N; j*=2) {
		pout = 1 - pout;
		pin = 1 - pout;

		if(i>=j) 
			buffer[i+N*pout]=buffer[i-j+N*pin]+buffer[i+N*pin];
		else
			buffer[i+N*pout]=buffer[i+N*pin];
		__syncthreads();
	}

	int d = img_size - min;

	//lut[i] = (cdf - min)*(nbr_bin - 1)/d;
	lut[i] = (int)(((float)buffer[pout*N+i] - min)*255/d + 0.5);
	if(lut[i] < 0){
		lut[i] = 0;
	}
}

__global__ void histogram_equalization_gpu(unsigned char * img_out, unsigned char * img_in, 
							int *lut) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	/* Construct the LUT by calculating the CDF */

	/* Get the result image */
	if(lut[img_in[i]] > 255){
		img_out[i] = 255;
	}
	else{
		img_out[i] = (unsigned char)lut[img_in[i]];
	}
}


