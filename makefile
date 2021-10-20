myPaint: main.o pgmProcess.o pgmUtility.o timing.o
	nvcc -o myPaint main.o pgmProcess.o pgmUtility.o timing.o

main.o: main.cu
	nvcc -c main.cu

pgmProcess.o: pgmProcess.cu
	nvcc -c pgmProcess.cu

pgmUtility.o: pgmUtility.c pgmUtility.h
	g++ -c -o pgmUtility.o pgmUtility.c -I.

timing.o: timing.c timing.h
	g++ -c timing.c -o timing.o -I.

clean:
	rm -r *.o myPaint 
