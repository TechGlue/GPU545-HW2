#include <stdio.h>
/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
__device__ float distance( int p1[], int p2[] ){
	return 0.0;
}

//idk why put this will only run with a global
//variables with d_ are input pointers and the _o is going to be our output array.
//this format will connect our main with the functions
__global__ void drawEdgeCUDA(int *d_pixels, char **d_header, int *o_pixels, int numRows, int numCols, int edgeWidth ){
        int col = blockIdx.x*blockDim.x + threadIdx.x;
        int row = blockIdx.y*blockDim.y + threadIdx.y;

        
}//end CUDA EDGE


//Both line and circle. Look at your headers and inputs before working.
__global__ void drawCircleCUDA(int *d_pixels, char **d_header, int *o_pixels, int numRows, int numCols, int centerRow, int centerCol, int radius){
        int col  = blockIdx.x*blockDim.x + threadIdx.x;
        int row   = blockIdx.y*blockDim.y + threadIdx.y;

}//end CUDACIRCLE

__global__ void drawLineCUDA(int *d_pixels, char **d_header, int *o_pixels, int numRows, int numCols, int p1row, int p1col, int p2row, int p2col){
        int Col  = blockIdx.x*blockDim.x + threadIdx.x;
        int row   = blockIdx.y*blockDim.y + threadIdx.y;

}//end CUDA LINE