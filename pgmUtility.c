//For our sequential code we implemented each circle, line, and edge in their respective pgm method below.
//All parallel code is all hosted in pgmProcess.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "pgmUtility.h"

// Implement or define each function prototypes listed in pgmUtility.h file.
// NOTE: Please follow the instructions stated in the write-up regarding the interface of the functions.
// NOTE: You might have to change the name of this file into pgmUtility.cu if needed.
/**
 *  Function Name:
 *      pgmRead()
 *      pgmRead() reads in a pgm image using file I/O, you have to follow the file format. All code > *
 *  @param[in,out]  header  holds the header of the pgm file in a 2D character array
 *                          After we process the pixels in the input image, we write the origianl
 *                          header (or potentially modified) back to a new image file.
 *  @param[in,out]  numRows describes how many rows of pixels in the image.
 *  @param[in,out]  numCols describe how many pixels in one row in the image.
 *  @param[in]      in      FILE pointer, points to an opened image file that we like to read in.
 *  @return         If successful, return all pixels in the pgm image, which is an int *, equivalent> * a 1D array that stores a 2D image in a linearized fashion. Otherwise null.
 *
 */
int * pgmRead(char ** header, int *numRows, int *numCols, FILE *in){
	char buff[50];
	int i = 0;
    
    while(i < 4) {
        fgets(buff, sizeof(buff), in);
        memcpy(header[i], buff, sizeof(buff));
        i++;
    }
	
    char * numRowColData = header[2];
	sscanf(numRowColData,"%d %d", numRows, numCols);

    size_t num_bytes = sizeof(int) * ((*numRows) * (*numCols));
	int * pixels = (int * ) malloc(num_bytes);

	for(i = 0; i < (*numRows) * (*numCols); i++){
		fscanf(in,"%d",&pixels[i]);
    }

    return pixels;
}

/** Sequential Solution
 *  Function Name:
 *      pgmDrawCircle()
 *      pgmDrawCircle() draw a circle on the image by setting relavant pixels to Zero.
 *                      In this function, you have to invoke a CUDA kernel to perform all image proc> *
 *  @param[in,out]  pixels  holds all pixels in the pgm image, which a 1D integer array. The array
 *                          are modified after the drawing.
 *  @param[in]      numRows describes how many rows of pixels in the image.
 *  @param[in]      numCols describes how many columns of pixels in one row in the image.
 *  @param[in]      centerCol specifies at which column you like to center your circle.
 *  @param[in]      centerRow specifies at which row you like to center your circle.
 *                        centerCol and centerRow defines the center of the circle.
 *  @param[in]      radius    specifies what the radius of the circle would be, in number of pixels.
 *  @param[in,out]  header returns the new header after draw.
 *                  the circle draw might change the maximum intensity value in the image, so we
 *                  have to change maximum intensity value in the header accordingly.
 *  @return         return 1 if max intensity is changed, otherwise return 0;
 */
int pgmDrawCircle( int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header ){
    
    if(pixels == NULL){
        printf("The array is empty. Exiting program...");
        exit(EXIT_FAILURE);
    }
    if(header == NULL){
        perror("The header is empty. Can't read in dimensions exiting program...");
        exit(EXIT_FAILURE);
    }

    int i, j;
    int setToZero = 0;

    for(i = 0; i < numRows; i++){
        for(j = 0; j < numCols; j++){
            if((pow(i - centerRow, 2)) + (pow(j - centerCol,2)) <= pow(radius,2)){
                pixels[(i*numCols)+j] = setToZero;
            }
        }
    }
    return 0;
}

/**     Sequential Solution 
 *      Function Name:
 *      pgmDrawEdge()
 *      pgmDrawEdge() draws a black edge frame around the image by setting relavant pixels to Zero.
 *                    In this function, you have to invoke a CUDA kernel to perform all image proces> *
 *  @param[in,out]  pixels  holds all pixels in the pgm image, which a 1D integer array. The array
 *                          are modified after the drawing.
 *  @param[in]      numRows describes how many rows of pixels in the image.
 *  @param[in]      numCols describes how many columns of pixels in one row in the image.
 *  @param[in]      edgeWidth specifies how wide the edge frame would be, in number of pixels.
 *  @param[in,out]  header returns the new header after draw.
 *                  the function might change the maximum intensity value in the image, so we
 *                  have to change the maximum intensity value in the header accordingly.
 *
 *  @return         return 1 if max intensity is changed by the drawing, otherwise return 0;
 */
int pgmDrawEdge( int *pixels, int numRows, int numCols, int edgeWidth, char **header ){
	int j, k, i;
	for(j = 0; j < numRows; j++) {
		for(k = 0; k < numCols; k++) {
            if(j < edgeWidth || j >= numRows-edgeWidth)
                pixels[j*numCols + k] = 0;
            else if(k < edgeWidth || k >= numCols-edgeWidth)
                pixels[j*numCols + k] = 0;
		}
	}
    return 0;
}

/** Sequential Solution
 *  Function Name:
 *      pgmDrawLine()
 *      pgmDrawLine() draws a straight line in the image by setting relavant pixels to Zero.
 *                      In this function, you have to invoke a CUDA kernel to perform all image processin> *
 *  @param[in,out]  pixels  holds all pixels in the pgm image, which a 1D integer array. The array
 *                          are modified after the drawing.
 *  @param[in]      numRows describes how many rows of pixels in the image.
 *  @param[in]      numCols describes how many columns of pixels in one row in the image.
 *  @param[in]      p1row specifies the row number of the start point of the line segment.
 *  @param[in]      p1col specifies the column number of the start point of the line segment.
 *  @param[in]      p2row specifies the row number of the end point of the line segment.
 *  @param[in]      p2col specifies the column number of the end point of the line segment.
 *  @param[in,out]  header returns the new header after draw.
 *                  the function might change the maximum intensity value in the image, so we
 *                  have to change the maximum intensity value in the header accordingly.
 *
 *  @return         return 1 if max intensity is changed by the drawing, otherwise return 0;
 */
int pgmDrawLine( int *pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col ){
    int diff_x = p2col - p1col;
	int diff_y = p2row - p1row;
	
	//Slope is calculated by (p2col - p1col)/(p2row - p1row)
	float slope = (float) diff_y / (float) diff_x;
	float yCur = (float) p1row;
	
	if(diff_x == 0){
		if(p1row < p2row){
			for(;p1row < p2row; p1row++)
				pixels[p1row * numCols + p1col] = 0;
		}else{
			for(;p1row > p2row; p2row--)
				pixels[p2row * numCols + p1col] = 0;
		}
	}else{
		if(p1col < p2col){
			for(p1col; p1col < p2col; p1col++){ 
			    pixels[(int) yCur * numCols + p1col] = 0;
			    yCur += slope;
			}
		}else{
			for(p1col; p1col > p2col; p1col--){
				pixels[(int) yCur * numCols + p1col] = 0;
				yCur += slope;
			}
		}
	}
    return 0;
}

/**
 *  Function Name:
 *      pgmWrite()
 *      pgmWrite() writes headers and pixels into a pgm image using file I/O.
 *                 writing back image has to strictly follow the image format. All code in this function > *
 *  @param[in]  header  holds the header of the pgm file in a 2D character array
 *                          we write the header back to a new image file on disk.
 *  @param[in]  pixels  holds all pixels in the pgm image, which a 1D integer array that stores a 2D imag> *  @param[in]  numRows describes how many rows of pixels in the image.
 *  @param[in]  numCols describe how many columns of pixels in one row in the image.
 *  @param[in]  out     FILE pointer, points to an opened text file that we like to write into.
 *  @return     return 0 if the function successfully writes the header and pixels into file.
 *                          else return -1;
 */
int pgmWrite( char **header, const int *pixels, int numRows, int numCols, FILE *out ){

    if(pixels == NULL)
    {
        perror("The passed in pixels is empty. Can't write out exiting program...");
        exit(EXIT_FAILURE);
    }
    if(header == NULL)
    {
        perror("The header is empty. Can't read in dimensions exiting program...");
        exit(EXIT_FAILURE);
    }

    int numPosition;
    int i, j; 

    //writing pgm file type
    fprintf(out, "%s", *header);

    //writing out comment for the image name
    fprintf(out, "%s", header[1]); 

    //writing out dimensions
    fprintf(out, "%s",header[2]);

    //writing out max gray located in row 3 of the header and turning it into a integer
    fprintf(out, "%s", header[3]);
    
    for(i = 0; i < numRows; i++)
    {
        for(j = 0; j<numCols; j++)
        {
            numPosition = pixels[(i*numCols)+j];
            fprintf(out, "%d ",numPosition);
        }
        fprintf(out, "\n");
    }
    
    return 0;
}

void usage()
{
        printf("Usage:\n    -e edgeWidth  oldImageFile  newImageFile\n    -c circleCenterRow circleCenterCol radius  oldImageFile  newImageFile\n    -l  p1row  p1col  p2row  p2col  oldImageFile  newImageFile\n");

}

ArgOption parseOpt(int argc, char *argv[])
{
    if(argc != 5 && argc != 7 && argc != 8)
    {
        return OPT_NULL;
    } 
    else
    {
        if(strlen( argv[1] ) != 2)
            return OPT_NULL;
    }

    int ch = (int)argv[1][1];

    if(ch < 97)
        ch = ch + 32;

    if (ch == 'c'){
        if(argc != 7)
            return OPT_NULL;
        return OPT_CIRCLE;
    }
      
    if (ch == 'l'){
        if(argc != 8)
            return OPT_NULL;
        return OPT_LINE;
    }

    if (ch == 'e'){
        if(argc != 5)
            return OPT_NULL;
        return OPT_EDGE;
    }

    return OPT_NULL;
}

void parseArgsCircle(char *argv[], int *circleCenterRow, int *circleCenterCol, 
                    int *radius, char originalImageName[], char newImageFileName[])
{
    *circleCenterRow = atoi(argv[2]);
    *circleCenterCol = atoi(argv[3]);
    *radius = atoi(argv[4]);
    strcpy(originalImageName, argv[5]);
    strcpy(newImageFileName, argv[6]);
    printf("%d - %d - %d - %s - %s\n", *circleCenterRow, *circleCenterCol, 
            *radius, originalImageName, newImageFileName);
}

void parseArgsLine(char *argv[], int *p1y, int *p1x, int *p2y, int *p2x, char originalImageName[], char newImageFileName[])
{
    *p1y = atoi(argv[2]);
    *p1x = atoi(argv[3]);

    *p2y = atoi(argv[4]);
    *p2x = atoi(argv[5]);


    strcpy(originalImageName, argv[6]);
    strcpy(newImageFileName, argv[7]);
	printf("%d - %d - %d - %d - %s - %s\n", *p1y, *p1x, *p2y, *p2x, originalImageName, newImageFileName);
}

void parseArgsEdge(char *argv[], int *edgeWidth, char originalImageName[], char newImageFileName[])
{
    *edgeWidth = atoi(argv[2]);
    strcpy(originalImageName, argv[3]);
    strcpy(newImageFileName, argv[4]);
	printf("%d - %s - %s\n", *edgeWidth, originalImageName, newImageFileName);
}