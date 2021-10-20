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
        int blockId = blockIdx.x + blockIdx.y * gridDim.x;
        int threadId = blockId * (blockDim.x * blockDim.y)
                + (threadIdx.y * blockDim.x) + threadIdx.x;

        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y; 

        if(row < edgeWidth || row > numRows - edgeWidth && threadId < numCols * numRows)
		o_pixels[threadId] = 0;
	else if(col < edgeWidth || col > numCols - edgeWidth && col < numCols && threadId < numCols * numRows)
		o_pixels[threadId] = 0;
        
}//end CUDA EDGE


//Both line and circle. Look at your headers and inputs before working.
__global__ void drawCircleCUDA(int *d_pixels, char **d_header, int *o_pixels, int numRows, int numCols, int centerRow, int centerCol, int radius){
        int blockId = blockIdx.x + blockIdx.y * gridDim.x;
        int threadId = blockId * (blockDim.x * blockDim.y)
                + (threadIdx.y * blockDim.x) + threadIdx.x;

        int col  = blockIdx.x*blockDim.x + threadIdx.x;
        int row  = blockIdx.y*blockDim.y + threadIdx.y;

        if(sqrtf(pow((double) row - centerRow, 2.0) + pow((double)col - centerCol, 2)) < radius)
                o_pixels[threadId] = 0;

}//end CUDACIRCLE

__global__ void drawLineCUDA(int *d_pixels, char **d_header, int *o_pixels, int numRows, int numCols, int p1row, int p1col, int p2row, int p2col){
                int blockId = blockIdx.x + blockIdx.y * gridDim.x;
        int threadId = blockId * (blockDim.x * blockDim.y)
                + (threadIdx.y * blockDim.x) + threadIdx.x;

        int col  = blockIdx.x*blockDim.x + threadIdx.x;
        int row  = blockIdx.y*blockDim.y + threadIdx.y;

        float slope = (float) (p2row - p1row) / (float) (p2col - p1col);
        float intercept = p1row - ((float) p1col) * slope; 
	if(isinf(slope)){
		if(row >= min(p1row,p2row) && row <= max(p1row,p2row) && col == p1col)
			o_pixels[threadId] = 0;
	}
        else if(abs(row - (float) (slope * col + intercept)) <= 1 && col <= max(p1col, p2col) && col >= min(p1col, p2col) && row <= max(p1row, p2row) && row >= min(p1row, p2row))
                o_pixels[threadId] = 0;

}//end CUDA LINE