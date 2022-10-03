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

1. Iterar sobre folder de imágenes

2. Cargar cada imagen

3. Redimencionar cada imagen

4. Calcular los gradientes

5. Dividir en celdas de 8x8

6. Calcular histogramas

7. Normalizar en bloques de 16 x 16

8. Pasar los histogramas normalizados por un clasificador (p.e. SVM)

9. ?

10. Profit

## Compiling instructions

To compile this program, I need to have cmake installed

    mkdir build
    cd build
    cmake ..
    make

It generates a `build/TireDetector` executable file, then I can use with

    build/TireDetector /path/to/images
