# Tire detector using HOG

This code is part of my project for the distributed systems subject on the master's degree in computer and systems engineering

More on HOG detectors:

https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

The algorith is based on the one showed in this article: https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f

## Steps

### Get dataset

1. Conseguir imágenes donde haya llantas

2. Marcar las imágenes con llantas

3. Extraer las imágenes solo de las llantas

### Code

1. Load image

2. Resize image in 64 x 64

3. Calculate gradients

4. Divide in 8x8 cells

5. Calculate histograms

6. Normalize 2 x 2 blocks

7. Train classifier (p.e. SVM)

8. Create a menu to:

a. Train

b. Test

c. Predict

9. ?

10. Profit

## Notes

For training, I need to input a folder path, it should generate a json file with training results. For testing I should pass a folder path and a json file with trained weights, and to predict i should pass an image path and a path for the json trained weights.

## Compiling instructions

To compile this program, I need to have cmake installed

    mkdir build
    cd build
    cmake ..
    make

It generates a `build/TireDetector` executable file, then I can use with

    build/TireDetector /path/to/images

