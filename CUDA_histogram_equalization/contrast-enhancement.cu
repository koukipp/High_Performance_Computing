#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

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

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
	PGM_IMG result, d0_result, d1_result;
	int *d0_hist, *d1_hist, *d0d1_hist, *d0_lut, *d1_lut, *d_luts[2], *h_hists[2];
	int maxBlocKSize = 1024;
	int i;
	int multi = 0;
	int **d_hists;
	unsigned char *d_results[2];
	int gpus = 2 - multi;

	int stream_number = 16;
	if(multi){
		stream_number = stream_number/2;
	}
	if(img_in.w * img_in.h < maxBlocKSize*stream_number){
		stream_number = 1;
		multi = 1;
	}
	cudaStream_t streams[stream_number];

	result.w = img_in.w;
	result.h = img_in.h;

	int transfer_size =  (img_in.w * img_in.h / (stream_number * maxBlocKSize)*maxBlocKSize);
	int transfer_size_mod = (img_in.w * img_in.h) % (stream_number * maxBlocKSize);

	struct timespec start, end; 
	double time_taken; 
	clock_gettime(CLOCK_MONOTONIC, &start);
	cudaSetDevice(0);
	gpuErrchk(cudaMalloc( (void**) &d0_lut, 256 * sizeof(int)));

	gpuErrchk(cudaMalloc( (void**) &d0_hist, 256 * sizeof(int)));
	gpuErrchk(cudaMalloc( (void**) &d0d1_hist, 256 * sizeof(int)));
	gpuErrchk(cudaMalloc( (void***) &d_hists, 2 * sizeof(int *)));

	if(!multi){
		gpuErrchk(cudaMalloc( (void**) &d0_result.img, 
		transfer_size * stream_number / gpus * sizeof(unsigned char)));
		cudaSetDevice(1); 
		gpuErrchk(cudaMalloc( (void**) &d1_lut, 256 * sizeof(int)));

		gpuErrchk(cudaMalloc( (void**) &d1_hist, 256 * sizeof(int)));
		gpuErrchk(cudaMalloc( (void**) &d1_result.img, transfer_size * stream_number / gpus * sizeof(unsigned char)
			+ transfer_size_mod * sizeof(unsigned char)));
	}
	else{
			gpuErrchk(cudaMalloc( (void**) &d0_result.img, 
		img_in.w * img_in.h * sizeof(unsigned char)));

	}

	gpuErrchk(cudaHostAlloc( (void**) &result.img, result.w * result.h * sizeof(unsigned char), 
		cudaHostAllocPortable));
	clock_gettime(CLOCK_MONOTONIC, &end);
	time_taken = (end.tv_sec - start.tv_sec) * 1e9; 
	time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-6; 
	printf("Malloc time %f\n", time_taken);
	
	d_results[0] = d0_result.img;
	d_luts[0] = d0_lut;
	h_hists[0] = d0_hist;

	if(!multi){
		d_results[1] = d1_result.img;
		h_hists[1] = d1_hist;
		d_luts[1] = d1_lut;
		cudaSetDevice(0);
	}

	dim3 grid(img_in.w * img_in.h / (maxBlocKSize * stream_number), 1, 1);
	dim3 block(maxBlocKSize, 1, 1);

	int temp = ((img_in.w * img_in.h) % (maxBlocKSize * stream_number))/maxBlocKSize;

	dim3 grid_mod(temp, 1, 1);
	dim3 block_mod(maxBlocKSize, 1, 1);

	dim3 grid_mod_mod(1, 1, 1);
	dim3 block_mod_mod(((img_in.w * img_in.h) % (maxBlocKSize * stream_number))%maxBlocKSize, 1, 1);

	clock_gettime(CLOCK_MONOTONIC, &start);
	gpuErrchk(cudaMemset(d0_hist, 0, 256 * sizeof(int)));

	if(!multi){
		cudaSetDevice(1);
		gpuErrchk(cudaMemset(d1_hist, 0, 256 * sizeof(int)));
	}

	int def_stream = 1 - multi;
	int boolean = 0;
	for(i=0; i<stream_number; i++) {
		cudaSetDevice(boolean);

		cudaStreamCreate(&streams[i]);

		gpuErrchk(cudaMemcpyAsync(&d_results[boolean][(i/gpus)*transfer_size], &img_in.img[i*transfer_size], 
			transfer_size * sizeof(unsigned char), cudaMemcpyHostToDevice, streams[i]));

		histogram_gpu<<<grid, block, 0, streams[i]>>>(h_hists[boolean], 
			&d_results[boolean][(i/gpus)*transfer_size]);

		boolean = !(boolean+multi);
	}

	gpuErrchk(cudaPeekAtLastError());

	if(grid_mod.x!=0) {
	
		gpuErrchk(cudaMemcpyAsync(&d_results[def_stream][(stream_number/gpus)*transfer_size],
			&img_in.img[(stream_number)*transfer_size], 
			transfer_size_mod * sizeof(unsigned char), cudaMemcpyHostToDevice, streams[def_stream]));
		
		histogram_gpu<<<grid_mod, block_mod, 0, streams[def_stream]>>>(h_hists[def_stream], 
			&d_results[def_stream][(stream_number/gpus) * maxBlocKSize * grid.x]);

		if(block_mod_mod.x!=0) {
			histogram_gpu<<<grid_mod_mod, block_mod_mod, 0, streams[def_stream]>>>(h_hists[def_stream], 
				&d_results[def_stream][(stream_number/gpus) * maxBlocKSize * grid.x + maxBlocKSize * grid_mod.x]);
		}
	}
	
	// grid.x = img_in.w * img_in.h / maxBlocKSize;
	// block.x = maxBlocKSize;

	// grid_mod.x = 1;
	// block_mod.x = (img_in.w * img_in.h) % maxBlocKSize;

	gpuErrchk(cudaPeekAtLastError());
	if(!multi){
		gpuErrchk(cudaDeviceSynchronize());
		cudaSetDevice(0);

		h_hists[1] = d0d1_hist;
		gpuErrchk(cudaMemcpy(d_hists, h_hists, 2 * sizeof(int *), cudaMemcpyHostToDevice));

		gpuErrchk(cudaMemcpyPeer(d0d1_hist, 0, d1_hist, 1, 256 * sizeof(int)));

		prefix_sum<<<1, 256>>>(d_hists, d0_lut, img_in.w*img_in.h);

		gpuErrchk(cudaPeekAtLastError());

		gpuErrchk(cudaMemcpyPeer(d1_lut, 1, d0_lut, 0, 256 * sizeof(int)));

		gpuErrchk(cudaDeviceSynchronize());
	}
	else{
		h_hists[1] = d0d1_hist;
		gpuErrchk(cudaMemcpy(d_hists, h_hists, 2 * sizeof(int *), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemset(d0d1_hist,0, 256 * sizeof(int)))
		prefix_sum<<<1, 256>>>(d_hists, d0_lut, img_in.w*img_in.h);
	}
	boolean = 0;
	for(i=0; i<stream_number; i++) {
		cudaSetDevice(boolean);

		histogram_equalization_gpu<<<grid, block, 0, streams[i]>>>(&d_results[boolean][(i/gpus)*transfer_size], 
			&d_results[boolean][(i/gpus)*transfer_size], d_luts[boolean]);

		gpuErrchk(cudaMemcpyAsync(&result.img[i*transfer_size], &d_results[boolean][(i/gpus)*transfer_size], 
			transfer_size * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[i]));

		boolean = !(boolean+multi);
	}

	gpuErrchk(cudaPeekAtLastError());

	if(grid_mod.x!=0) {
		histogram_equalization_gpu<<<grid_mod, block_mod, 0, streams[def_stream]>>>(
			&d_results[def_stream][maxBlocKSize * grid.x * (stream_number/gpus)], 
			&d_results[def_stream][maxBlocKSize * grid.x * (stream_number/gpus)], d_luts[def_stream]);

		if(block_mod_mod.x!=0) {
			histogram_equalization_gpu<<<grid_mod_mod, block_mod_mod, 0, streams[def_stream]>>>(
				&d_results[def_stream][maxBlocKSize * grid.x * (stream_number/gpus) + maxBlocKSize*grid_mod.x], 
				&d_results[def_stream][maxBlocKSize * grid.x * (stream_number/gpus) + maxBlocKSize*grid_mod.x], d_luts[def_stream]);
		}

		gpuErrchk(cudaMemcpyAsync(&result.img[(stream_number)*transfer_size], 
			&d_results[def_stream][(stream_number/gpus)*transfer_size], 
			(grid_mod.x * maxBlocKSize + block_mod_mod.x) * sizeof(unsigned char), 
			cudaMemcpyDeviceToHost, streams[def_stream]));
	}

	gpuErrchk(cudaPeekAtLastError());

	for(i=0; i<gpus; i++) {
		cudaSetDevice(i);
		gpuErrchk(cudaDeviceSynchronize());
	}

	

	clock_gettime(CLOCK_MONOTONIC, &end); 
	time_taken = (end.tv_sec - start.tv_sec) * 1e9; 
	time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-6; 
	printf("GPU kernel exec+transfer time %f\n", time_taken);

	clock_gettime(CLOCK_MONOTONIC, &start);
	boolean = 0;
	for(i=0; i<stream_number; i++) {
		cudaStreamDestroy(streams[boolean]);
		boolean = !(boolean+multi);
	}

	if(!multi){
		cudaSetDevice(1);
		cudaFree(d1_lut);
		cudaFree(d1_hist);
		cudaFree(d1_result.img);

		cudaSetDevice(0);
	}

	cudaFree(d0_lut);
	cudaFree(d0_hist);
	cudaFree(d0d1_hist);
	cudaFree(d_hists);
	cudaFree(d0_result.img);
	clock_gettime(CLOCK_MONOTONIC, &end); 
	time_taken = (end.tv_sec - start.tv_sec) * 1e9; 
	time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-6; 
	printf("Free-Destroy time %f\n", time_taken);


	return result;
}
