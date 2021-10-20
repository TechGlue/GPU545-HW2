#include <stdio.h>
#include "pgmProcess.h"
#include "pgmUtility.h"
#include "timing.h"
int main(int argc, char *argv[]){

	double now, then, cost;
    ArgOption opt;
	FILE *fp = NULL, *out = NULL;
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

    int m, n, l, x, ch;
    int edgeWidth, circleCenterRow, circleCenterCol, radius;
    char originalImageName[100], newImageFileName[100];

    opt = parseOpt(argc, argv);

    //block of if's for parsing the input for each specific shape type. Once parsed the content of the variables will be changed 
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

    //GPU METHOD SET_UP
    
    //input data 1d array and 2d array 
    int * d_pixels;
    char ** d_header;

    //output 1D array
    int *o_pixels;
    //number of bytes for the two variables from above. 
    size_t bytes = (sizeof(int) * (numRows * numCols));
    size_t hbytes = (sizeof(char) * maxSizeHeadRow);

    then = currentTime();
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

    
    //The actuall logic methods that will help create the different shapes on the images.  
    if (opt == OPT_CIRCLE){
        drawCircleCUDA<<<gridSize2D, blockSize2D>>>(d_pixels,d_header,o_pixels,numRows, numCols, circleCenterRow, circleCenterCol, radius);
    }
    if (opt == OPT_EDGE) {
        drawEdgeCUDA<<<gridSize2D, blockSize2D>>>(d_pixels,d_header,o_pixels,numRows,numCols,edgeWidth);

    }
    if (opt == OPT_LINE){
        //pgmDrawLine(pixels, numRows, numCols, header, p1y, p1x, p2y, p2x);
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
    now = currentTime();
	cost = now - then;	
    printf("Code execution time: %lf\n", cost);

    //once we've done our echanges we are going to pass our one d array and print it out as a 2D-array 
    pgmWrite(header, pixels, numRows, numCols, out );

    //free cuda memory
	cudaFree(d_pixels);
	cudaFree(d_header);
    cudaFree(d_header);

    i = 0;
    for(;i < rowsInHeader; i++)
        free(header[i]);
    free(header);
    if(out != NULL)
        fclose(out);
    if(fp != NULL)
        fclose(fp);

	return 0;
}