#include <stdio.h>
#include "pgmProcess.h"
#include "pgmUtility.h"
//wrote some notes to help me understand - lg

int main(int argc, char *argv[]){

    ArgOption opt;
	FILE *fp = NULL, *out = NULL;
	char ** header = (char**) malloc( sizeof(char *) * rowsInHeader);

    int i;
    int * pixels = NULL;

    for(i = 0; i < 4; i++){
        header[i] = (char *) malloc (sizeof(char) * maxSizeHeadRow);
    }

    int numRows, numCols;

    int p1y = 0;
    int p1x = 0;
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
        usage();
        return 1;
    }

    //NOTE THE 1D ARRAY LOOKING GOOD IT HAS THE
    //Reading in the actual pgm file.
    pixels = pgmRead(header, &numRows, &numCols, fp);

	//declare device memory for pixels and header
        int * d_pixels;
        char ** d_header;
        size_t bytes = (sizeof(int) * (numRows * numCols));
        cudaMalloc(&d_pixels, bytes);

        //header bytes
        size_t hbytes = (sizeof(char) * maxSizeHeadRow);
        cudaMalloc(&d_header, hbytes);

	//cudaMemCopys for pixels/headers
	cudaMemcpy(d_pixels, pixels, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_header, header, hbytes, cudaMemcpyHostToDevice);
	//not sure what to do for grid size or n so we're gonna do 100000 like vecAdd example
	int n1 = 100000, blockSize = 1024, gridSize;
	gridSize = (int)ceil((float)n1/blockSize);

    //The actuall logic methods that will help create the different shapes on the images.  
    if (opt == OPT_CIRCLE)
        //pgmDrawCircle(pixels, numRows, numCols, circleCenterRow, circleCenterCol, radius, header );
        //UNCOMMENT THE LINE BELOW AND COMMENT THE LINE ABOVE TO RUN 
        pgmDrawCircleCPU(pixels, numRows, numCols, circleCenterRow, circleCenterCol, radius, header );
    if (opt == OPT_EDGE) {
	//declare device memories needed for edge
        pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);
	//cudaDrawEdge<<<gridSize, blockSize>>>(d_pixels, numRows, numCols, edgeWidth, d_header);

	}
    if (opt == OPT_LINE)
        pgmDrawLine(pixels, numRows, numCols, header, p1y, p1x, p2y, p2x);

	//cuda memcpy back to host
	cudaMemcpy(pixels, d_pixels, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(header, d_header, hbytes, cudaMemcpyDeviceToHost);
    //once we've done our echanges we are going to pass our one d array and print it out as a 2D-array 
    pgmWrite(header, pixels, numRows, numCols, out );

	//free cuda mems
	cudaFree(d_pixels);
	cudaFree(d_header);


    i = 0;
    //freeing the numbers was behaving weird so commented out just to compile
    //for(;i < 512 * 512; i++)
	//free(pixels[i]);
    //deallocateArray(pixels, numCols, numRows);
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
