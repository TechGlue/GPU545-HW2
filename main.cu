/*
No extra credit attempted 
HW2 Group Project Members
Ryley Horton
Luis Garcia
Evan Bell
Carl Painter
*/
#include <stdio.h>
#include "pgmProcess.h"
#include "pgmUtility.h"
#include "timing.h"
int main(int argc, char *argv[]){

	double cpuNow, cpuThen, cpuCost;
	double gpuNow, gpuThen, gpuCost;
    ArgOption opt;
	FILE *fp = NULL, *out = NULL;
    FILE *efficiencyOut = fopen("TimeAndSpeedup.txt", "w");
	char ** header = (char**) malloc( sizeof(char *) * rowsInHeader);

    int i;
    int * pixels = NULL;

    for(i = 0; i < 4; i++){
        header[i] = (char *) malloc (sizeof(char) * maxSizeHeadRow);
    }

    int numRows, numCols;

    int p1y = 0; // row
    int p1x = 0; // col
    int p2y = 0;
    int p2x = 0;

    int edgeWidth, circleCenterRow, circleCenterCol, radius;
    char originalImageName[100], newImageFileName[100];

    opt = parseOpt(argc, argv);

    if (opt == OPT_CIRCLE)
        parseArgsCircle(argv, &circleCenterRow, &circleCenterCol, &radius, originalImageName, newImageFileName);

    if (opt == OPT_EDGE)
        parseArgsEdge(argv, &edgeWidth, originalImageName, newImageFileName);

    if (opt == OPT_LINE)
        parseArgsLine(argv, &p1y, &p1x, &p2y, &p2x, originalImageName, newImageFileName);

    if (opt != OPT_NULL){
        fp = fopen(originalImageName, "r");
        out = fopen(newImageFileName, "w");
    }

    if(fp == NULL || out == NULL || opt == OPT_NULL){
        if(fp != NULL){
            printf("Input file pointer is null, closing.");
            fclose(fp);
            return 1;
        } 
        if(out != NULL){
            printf("Output file pointer is null, closing.");
            fclose(out);
            return 1;
        } 
        printf("No appropriate option specified in arguments.");
        usage();
        return 1;
    }

    pixels = pgmRead(header, &numRows, &numCols, fp);

    //input data 1d array and 2d array 
    int * d_pixels;
    char ** d_header;

    //output 1D array
    int *o_pixels;
    //number of bytes for the two variables from above. 
    size_t bytes = (sizeof(int) * (numRows * numCols));
    size_t hbytes = (sizeof(char) * maxSizeHeadRow);
    
    cudaMalloc(&d_header, hbytes);
    cudaMalloc(&d_pixels, bytes);
    cudaMalloc(&o_pixels, bytes);

    //copying host to the device. Pixels being copied to d_pixels. Same for host.
	cudaMemcpy(d_pixels, pixels, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_header, header, hbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(o_pixels, pixels, bytes, cudaMemcpyHostToDevice);
	
    int threadsPerBlock = 32; 

    int gridDimX = ceil(numRows / (double) threadsPerBlock);
    int gridDimY = ceil(numCols);
    printf("Launching with grid of dimensions (%d, %d)\n", gridDimX, gridDimY);
    dim3 gridSize2D(gridDimX, gridDimY);
    dim3 blockSize2D(threadsPerBlock);
    
    if (opt == OPT_CIRCLE){
    	cpuThen = currentTime();
    	pgmDrawCircle(pixels, numRows, numCols, circleCenterRow, circleCenterCol, radius, header);
    	cpuNow = currentTime();
        //Start GPU
    	gpuThen = currentTime();
        drawCircleCUDA<<<gridSize2D, blockSize2D>>>(d_pixels,d_header,o_pixels,numRows, numCols, circleCenterRow, circleCenterCol, radius);
    }
    if (opt == OPT_EDGE) {
    	cpuThen = currentTime();
    	pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);
    	cpuNow = currentTime();
    	//Start GPU
    	gpuThen = currentTime();
        drawEdgeCUDA<<<gridSize2D, blockSize2D>>>(d_pixels,d_header,o_pixels,numRows,numCols,edgeWidth);
    }
    if (opt == OPT_LINE){
    	cpuThen = currentTime();
    	pgmDrawLine(pixels, numRows, numCols, header, p1y, p1x, p2y, p2x);
    	cpuNow = currentTime();
    	//Start GPU
    	gpuThen = currentTime();
	    drawLineCUDA<<<gridSize2D, blockSize2D>>>(d_pixels, d_header, o_pixels, numRows, numCols, p1y, p1x, p2y, p2x);
    }
    
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       
    }

    //cuda memcpy back to host
	cudaMemcpy(pixels, o_pixels, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(header, d_header, hbytes, cudaMemcpyDeviceToHost);
    
    //timing 
    gpuNow = currentTime();
    cpuCost = cpuNow - cpuThen;
    gpuCost = gpuNow - gpuThen;
    
    fprintf(efficiencyOut,"***Time Cost and Speed Up***\nCPU Execution Time: %lf\nGPU Execution Time: %lf\nSpeedup: %lf\nEfficiency:%lf\n",cpuCost,gpuCost,cpuCost/gpuCost,cpuCost/gpuCost/threadsPerBlock);
    printf("\nCPU Execution Time: %lf\nGPU Execution Time: %lf\nSpeedup: %lf\nEfficiency:%lf\n",cpuCost,gpuCost,cpuCost/gpuCost,cpuCost/gpuCost/threadsPerBlock);
    pgmWrite(header, pixels, numRows, numCols, out );

    //free cuda memory
	cudaFree(d_pixels);
	cudaFree(d_header);
    cudaFree(d_header);

    i = 0;
    for(;i < 4; ++i)
        free(header[i]);
    free(header);
    printf("Closing out file...\n");
    if(out != NULL)
        fclose(out);
    printf("Closing fp file...\n");
    if(fp != NULL)
        fclose(fp);
    printf("Closing efficiency out file...\n");
    if(efficiencyOut != NULL)
        fclose(efficiencyOut);

	return 0;
}
