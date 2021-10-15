#include <stdio.h>
#include "pgmProcess.h"
#include "pgmUtility.h"

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

    pixels = pgmRead(header, &numRows, &numCols, fp);

    if (opt == OPT_CIRCLE)
        pgmDrawCircle(pixels, numRows, numCols, circleCenterRow, circleCenterCol, radius, header );
                    pgmWrite(header,pixels, numRows, numCols, out );    
    if (opt == OPT_EDGE)
        pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);
    
    if (opt == OPT_LINE)
        pgmDrawLine(pixels, numRows, numCols, header, p1y, p1x, p2y, p2x);
                
                
    pgmWrite(header, pixels, numRows, numCols, out );


    i = 0;
//freeing the numbers was behaving weird so commented out just to compile
/*    for(;i < numRows; i++)
        free(pixels[i]);
    free(pixels);
  */  i = 0;
    for(;i < rowsInHeader; i++)
        free(header[i]);
    free(header);
    if(out != NULL)
        fclose(out);
    if(fp != NULL)
        fclose(fp);

	return 0;
}