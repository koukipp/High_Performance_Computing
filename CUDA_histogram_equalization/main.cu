#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <time.h>

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

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename);

int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;


    struct timespec start, end; 
	double time_taken; 


	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
	
	printf("Running contrast enhancement for gray-scale images.\n");
    clock_gettime(CLOCK_MONOTONIC, &start); 
	img_ibuf_g = read_pgm(argv[1]);
	run_cpu_gray_test(img_ibuf_g, argv[2]);
    free_pgm(img_ibuf_g);
    clock_gettime(CLOCK_MONOTONIC, &end); 
	time_taken = (end.tv_sec - start.tv_sec) * 1e9; 
	time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-6; 
	printf("Total execution time %f\n", time_taken);

	return 0;
}



void run_cpu_gray_test(PGM_IMG img_in, char *out_filename)
{
    PGM_IMG img_obuf;

    printf("Starting GPU processing...\n");
    img_obuf = contrast_enhancement_g(img_in);
    write_pgm(img_obuf, out_filename);
    free_pgm(img_obuf);
}


PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    
    struct timespec start, end; 
	double time_taken; 
    clock_gettime(CLOCK_MONOTONIC, &start); 
    
    gpuErrchk(cudaHostAlloc( (void**) &result.img, result.w * result.h * sizeof(unsigned char), 
        cudaHostAllocPortable));
    
    clock_gettime(CLOCK_MONOTONIC, &end); 
	time_taken = (end.tv_sec - start.tv_sec) * 1e9; 
	time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-6; 
	printf("Img_in malloc time %f\n", time_taken);

    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;

    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    cudaFreeHost(img.img);
}




