myPaint: main.o pgmProcess.o pgmUtility.o timing.o
	nvcc -arch=sm_61 -o myPaint main.o pgmProcess.o pgmUtility.o timing.o

main.o: main.cu
	nvcc -arch=sm_61 -c main.cu

pgmProcess.o: pgmProcess.cu
	nvcc -arch=sm_61 -c pgmProcess.cu

pgmUtility.o: pgmUtility.c pgmUtility.h
	g++ -c -o pgmUtility.o pgmUtility.c -I.

timing.o: timing.c timing.h
	g++ -c timing.c -o timing.o -I.

clean:
	rm -r *.o myPaint 
