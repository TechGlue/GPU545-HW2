# GPU545 Homework 2 Simple Image Processing on CUDA Device

This repository is the home to a group assignment for CSCD 545 GPU Computing. In this project we manipulate PGM images (Portable Grap Map). What PGM images are a file that is made up of a header with information on the image (P2 tells us it's an ascii pgm file, name of file, dimensions of the grid, and the maximum intensity of each pixel) and a grid of different brightness intensities. 

**Our task/goal was to utilize both CPU and GPU to manipulate these pgm files by drawing shapes on them and then compare how well the program performs.**

| **Team**       |
| -------------- |
| Luis Garcia    |
| Carl Painter   |
| Ryley Horton   |
| Evan Bell      |

## Technologies
The project is utilizing:

- **C** 
- **CUDA C**

## Deployment / Usage 

To deploy this project run. 

First clone the project to your local machine

```bash
  git clone https://github.com/TechGlue/GPU545-HW2.git
```

Navigate to the project folder.
```bash
  cd GPU545-HW2
```
Clean and compile the project.
```bash
  make clean
  make
```
**Usage:**
 
**-e edgeWidth  oldImageFile  newImageFile**

**-c circleCenterRow circleCenterCol radius  oldImageFile  newImageFile**

**-l p1row  p1col  p2row  p2col  oldImageFile  newImageFile**

**Test Cases** 
```bash
  ./myPaint -c 470 355 100  ./balloons.ascii.pgm  balloons_c100_4.pgm 
  ./myPaint -c 228 285 75  ./balloons.ascii.pgm  balloons_c75_5.pgm 
  
  ./myPaint -e 50  ./balloons.ascii.pgm  balloons_e50_2.pgm
  
  ./myPaint -l  1 5 50 200  ./balloons.ascii.pgm  balloons_l1.pgm
  ./myPaint -l  1 50 479 639  ./balloons.ascii.pgm  balloons_l2.pgm
  ./myPaint -l  1 50 479 639  ./balloons.ascii.pgm  balloons_l2.pgm
  ./myPaint –l  5 320 240 320  ./balloons.ascii.pgm  balloons_l4.pgm
```

## Demo

![Project Demo](./../ContributingGuide/GPUHW2Demo.gif?raw=true)
