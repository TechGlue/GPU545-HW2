#include <stdio.h>
#include "pgmProcess.h"
#include "pgmUtility.h"
#include "timing.h"
//wrote some notes to help me understand - lg

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
        if(fp != NULL) fclose(fp);
        if(out != NULL) fclose(out);
        usage();
        return 1;
    }

    then = currentTime();

    //NOTE THE 1D ARRAY LOOKING GOOD IT HAS THE
    //Reading in the actual pgm file. 
    pixels = pgmRead(header, &numRows, &numCols, fp);
    
    //The actuall logic methods that will help create the different shapes on the images.  
    if (opt == OPT_CIRCLE)
        pgmDrawCircle(pixels, numRows, numCols, circleCenterRow, circleCenterCol, radius, header );
        //UNCOMMENT THE LINE BELOW AND COMMENT THE LINE ABOVE TO RUN 
        //pgmDrawCircleCPU(pixels, numRows, numCols, circleCenterRow, circleCenterCol, radius, header );
    if (opt == OPT_EDGE) {
        pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);

	}
    if (opt == OPT_LINE)
        pgmDrawLine(pixels, numRows, numCols, header, p1y, p1x, p2y, p2x);
                
    now = currentTime();
	cost = now - then;	
    printf("Code execution time: %lf\n", cost);
    
    //once we've done our echanges we are going to pass our one d array and print it out as a 2D-array 
    pgmWrite(header, pixels, numRows, numCols, out );

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
