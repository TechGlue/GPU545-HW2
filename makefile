myPaint: main.o pgmProcess.o pgmUtility.o
	nvcc -arch=sm_52 -o myPaint main.o pgmProcess.o pgmUtility.o

main.o: main.cu
	nvcc -arch=sm_52 -c main.cu

pgmProcess.o: pgmProcess.cu
	nvcc -arch=sm_52 -c pgmProcess.cu

pgmUtility.o: pgmUtility.c pgmUtility.h
	g++ -c -o pgmUtility.o pgmUtility.c -I.

clean:
	rm -r *.o myPaint 
