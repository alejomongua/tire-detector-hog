#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include <omp.h>

#include "listDir.hpp"

using namespace std;
using namespace cv;

#define EPSILON 1e-10
#define PI 3.1416
#define FEATURE_VECTOR_SIZE 1764 // = 7 * 7 * 36
#define EULER 2.71828
#define EPOCHS 100
#define WEIGHTS_FILE_PATH "weights.data"
#define X_DIM 64
#define Y_DIM 64
#define CELL_SIZE 8
#define BINS 9
#define CELLS_X X_DIM / CELL_SIZE
#define CELLS_Y Y_DIM / CELL_SIZE

#ifndef NUMBER_OF_BLOCKS
#define NUMBER_OF_BLOCKS 6
#endif

#ifndef NUMBER_OF_THREADS
#define NUMBER_OF_THREADS 128
#endif

#ifndef OMP_NUM_THREADS
#define OMP_NUM_THREADS 4
#endif

const float cosine[9] = {
    1,
    0.939692,
    0.766043,
    0.499998,
    0.173645,
    -0.173652,
    -0.500004,
    -0.766048,
    -0.939695};
const float sine[9] = {
    0,
    -0.342021,
    -0.642789,
    -0.866027,
    -0.984808,
    -0.984807,
    -0.866023,
    -0.642783,
    -0.342014,
};

inline void checkCuda(cudaError_t err, const char *message)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s (error: %s)!\n", message, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

__constant__ float Cj[10];

__global__ void computeHistograms(float *angle, float *mag, float *features, int totalExamples)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    for (int _i = index; _i < totalExamples; _i += NUMBER_OF_BLOCKS * NUMBER_OF_THREADS)
    {
        float histogram[CELLS_X][CELLS_Y][BINS];
        // Process 8 x 8 blocks
        for (int i = 0; i < CELLS_X; i++)
        {
            for (int j = 0; j < CELLS_Y; j++)
            {
                // Initialize histogram:
                for (int k = 0; k < BINS; k++)
                {
                    histogram[i][j][k] = 0.0;
                }

                // Fill histogram
                for (int k = 0; k < 8; k++)
                {
                    for (int l = 0; l < 8; l++)
                    {
                        unsigned int coordX = k + i * CELL_SIZE;
                        unsigned int coordY = l + j * CELL_SIZE;
                        unsigned int index = _i * X_DIM * Y_DIM + coordX * Y_DIM + coordY;
                        float singleAngle = angle[index];
                        float singleMagnitude = mag[index];
                        if (singleAngle >= 180.0)
                        {
                            singleAngle = singleAngle - 180;
                            singleMagnitude = -singleMagnitude;
                        }
                        unsigned char valueJ = (unsigned char)(singleAngle / 20);
                        float vJ = singleMagnitude * (singleAngle - Cj[valueJ]) / 20;
                        float vJp1 = singleMagnitude - vJ;
                        histogram[i][j][valueJ] += vJp1;
                        // If valueJ is the last index, it should be 0 and change its sign
                        if (valueJ == 8)
                        {
                            histogram[i][j][0] -= vJ;
                        }
                        else
                        {
                            histogram[i][j][(valueJ + 1) % BINS] += vJ;
                        }
                    }
                }
            }
        }

        // Normalize
        for (int i = 0; i < CELLS_X - 1; i++)
        {
            for (int j = 0; j < CELLS_Y - 1; j++)
            {
                // baseIndex needs to be calculated only once
                unsigned int baseIndex = _i * FEATURE_VECTOR_SIZE + i * (CELLS_Y - 1) * BINS * 4 + j * BINS * 4;
                float powerSum = 0.0;
                for (int m = 0; m < BINS; m++)
                {
                    // Square root of sum of squares
                    powerSum += pow(histogram[i][j][m], 2) +
                                pow(histogram[i + 1][j][m], 2) +
                                pow(histogram[i][j + 1][m], 2) +
                                pow(histogram[i + 1][j + 1][m], 2);
                }
                float norm = sqrt(powerSum);

                for (int m = 0; m < BINS; m++)
                {
                    // EPSILON IS A SMALL NUMBER TO AVOID DIVISION BY 0
                    // 36 * 7 = 252
                    features[baseIndex + m * 4] =
                        histogram[i][j][m] / (norm + EPSILON);
                    features[baseIndex + m * 4 + 1] =
                        histogram[i + 1][j][m] / (norm + EPSILON);
                    features[baseIndex + m * 4 + 2] =
                        histogram[i][j + 1][m] / (norm + EPSILON);
                    features[baseIndex + m * 4 + 3] =
                        histogram[i + 1][j + 1][m] / (norm + EPSILON);
                }
            }
        }
    }
}

void preprocessImage(string path, float *magVector, float *anglesVector, int index)
{
    /*
     * Esta imagen convierte una ruta a un archivo en un array plano de float
     * para unirlo a un array plano más grande con todas las magnitudes y ángulos
     */
    Mat img, resizedImg, gx, gy, mag, angle;

    // Load images in an opencv matrix in gray scale
    img = imread(path, IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cerr << "Could not read the image: " << path << endl;
        std::exit(-1);
    }

    // Resize image
    // resize(img, resizedImg, Size(896, 896));
    // imshow("original", resizedImg);

    // Resize image
    resize(img, resizedImg, Size(X_DIM, Y_DIM));
    img.release();

    // Calculate gradients
    cv::Sobel(resizedImg, gx, CV_32F, 1, 0, 1);
    cv::Sobel(resizedImg, gy, CV_32F, 0, 1, 1);
    resizedImg.release();

    // Convert to polar
    cv::cartToPolar(gx, gy, mag, angle, 1);
    gx.release();
    gy.release();

    float *locationMag = &magVector[X_DIM * Y_DIM * index];
    float *locationAngle = &anglesVector[X_DIM * Y_DIM * index];

    memcpy(locationMag, (float *)mag.data, X_DIM * Y_DIM * sizeof(float));
    memcpy(locationAngle, (float *)angle.data, X_DIM * Y_DIM * sizeof(float));
    mag.release();
    angle.release();
}

void getFeatureVector(float *h_magVectors, float *h_anglesVectors, float *h_features, int totalExamples)
{
    // Allocate memory
    float *d_angle = NULL;
    float *d_mag = NULL;
    float *d_features = NULL;
    int dims = X_DIM * Y_DIM * totalExamples * sizeof(float);
    int fVectorDims = FEATURE_VECTOR_SIZE * totalExamples * sizeof(float);

    checkCuda(
        cudaMalloc(
            (void **)&d_angle, dims),
        "Failed to allocate device vector angles");

    checkCuda(
        cudaMalloc(
            (void **)&d_mag, dims),
        "Failed to allocate device vector magnitude");

    checkCuda(
        cudaMalloc(
            (void **)&d_features, fVectorDims),
        "Failed to allocate device feature vector");

    // Pass parameters
    checkCuda(
        cudaMemcpy(
            d_angle,
            h_anglesVectors, dims, cudaMemcpyHostToDevice),
        "Failed to copy vector angles from host ");

    checkCuda(
        cudaMemcpy(
            d_mag,
            h_magVectors, dims, cudaMemcpyHostToDevice),
        "Failed to copy vector magnitudes from host");

    // Launch kernel
    computeHistograms<<<NUMBER_OF_BLOCKS, NUMBER_OF_THREADS>>>(d_angle, d_mag, d_features, totalExamples);

    checkCuda(cudaGetLastError(), "Failed to launch kernel");

    checkCuda(
        cudaMemcpy(
            h_features, d_features, fVectorDims, cudaMemcpyDeviceToHost),
        "Failed to copy features from device to host");

    // Free device global memory
    checkCuda(cudaFree(d_angle), "Failed to free device vector angles");

    // Free device global memory
    checkCuda(cudaFree(d_mag), "Failed to free device vector magnitudes");

    // Free device global memory
    checkCuda(cudaFree(d_features), "Failed to free device features");
}

float predict(float *featureVector, float *weights)
{
    float sum = weights[0];

    for (int i = 0; i < FEATURE_VECTOR_SIZE; i++)
    {
        sum += weights[i + 1] * featureVector[i];
    }

    return 1 / (1 + pow(EULER, -sum));
}

float cost(unsigned char label, float prediction)
{
    if (prediction == label)
        return 0;
    float cost1 = label * log(prediction);
    float cost2 = (1 - label) * log(1 - prediction);
    return -cost1 - cost2;
}

void trainLogRegression(unsigned int epochs, unsigned int examples, float *features,
                        unsigned char *labels, float *weights)
{
    unsigned int i, j, k;
    float prediction, error, slope;
    float alpha = 0.001; // learning rate
    for (i = 0; i < FEATURE_VECTOR_SIZE + 1; i++)
    {
        weights[i] = 0;
    }

    for (i = 0; i < epochs; i++)
    {
        if (!(i % 100))
        {
            printf("Epoch %d/%d...\n", i, epochs);
        }

        // float costo = 0;
        for (j = 0; j < examples; j++)
        {
            prediction = predict(&features[j * FEATURE_VECTOR_SIZE], weights);
            error = labels[j] - prediction;
            slope = alpha * error;
            // cout << "Error=" << error << endl;
            weights[0] = weights[0] + slope;
            for (k = 1; k < FEATURE_VECTOR_SIZE + 1; k++)
            {
                weights[k] = weights[k] + slope * features[j * FEATURE_VECTOR_SIZE + k - 1];
            }
            // costo += cost(labels[j], prediction);
        }
        // cout << "Costo=" << costo / examples << endl;
    }
}

/*
 * This function loads weights from weights file
 * it returns 1 if there are no weights or 0 if
 * weights are loaded
 */
unsigned char loadWeights(float *weights)
{
    char floatToStr[20];
    unsigned int i = 0, j = 0;
    ifstream weightsInFile(WEIGHTS_FILE_PATH);

    if (!weightsInFile.good())
    {
        return 1;
    }
    std::vector<int> load;
    char temp;

    while (weightsInFile.read(&temp, 1))
    {
        floatToStr[i++] = temp;
        if (temp == ' ')
        {
            floatToStr[i] = 0;
            weights[j++] = atof(floatToStr);
            i = 0;
        }
    }

    weightsInFile.close();
    return 0;
}

void drawVector(Mat *img, float *featureVector, unsigned int baseIndex, unsigned char index)
{
    const int width = 64;
    *img = Mat::zeros(width, width, CV_32F);
    for (int k = 0; k < 9; k++)
    {
        unsigned int x = (unsigned int)(featureVector[baseIndex + k * 4 + index] * cosine[k] * width / 2 + width / 2);
        unsigned int y = (unsigned int)(featureVector[baseIndex + k * 4 + index] * sine[k] * width / 2 + width / 2);

        circle(
            *img,
            Point(width / 2, width / 2),
            2,
            Scalar(255),
            2,
            1,
            0);
        line(
            *img,
            Point(width / 2, width / 2),
            Point(x, y),
            Scalar(255),
            1,
            LINE_8);
    }
}
void drawFeatureVector(float *featureVector)
{
    unsigned int i, j, k;
    Mat img, img1, img2, img3[4], img4, img5;

    for (i = 0; i < (CELLS_X - 1); i++)
    {
        for (j = 0; j < (CELLS_Y - 1); j++)
        {
            for (k = 0; k < 4; k++)
            {
                drawVector(&img3[k], featureVector, i * (CELLS_Y - 1) * BINS * 4 + j * BINS * 4, k);
            }
            hconcat(img3[0], img3[1], img4);
            hconcat(img3[2], img3[3], img5);

            if (j)
            {
                vconcat(img4, img5, img1);
                hconcat(img, img1, img);
            }
            else
            {
                vconcat(img4, img5, img);
            }
        }
        if (img2.empty())
        {
            img2 = img.clone();
        }
        else
        {
            vconcat(img2, img, img2);
        }
    }

    imshow("Display window", img2);
    cv::waitKey(0); // Wait for a keystroke in the window
}

int main(int argc, const char **argv)
{
    const char *nonTiresPath;
    const char *tiresPath;
    vector<string> tireImagePaths, noTireImagePaths;
    int i, epochs;
    unsigned int tireImagesVectorSize, noTireImagesVectorSize, totalSize;
    float weights[FEATURE_VECTOR_SIZE + 1];
    float prediction;
    float *magVectors;
    float *anglesVectors;
    float *features;
    unsigned char *labels;
    char floatToStr[20];
    float h_Cj[10];

    for (i = 0; i < 10; i++)
    {
        h_Cj[i] = 20.0 * i;
    }

    // copy Cj to GPU
    checkCuda(
        cudaMemcpyToSymbol(Cj, &h_Cj[0], 10 * sizeof(float), size_t(0), cudaMemcpyHostToDevice),
        "Failed to copy Cj to device");

    // If two paths are passed, it should train
    // the first path is the folder with tires, and the second path is the
    // non-tires folder
    if (argc > 2)
    {
        tiresPath = argv[1];
        nonTiresPath = argv[2];
        epochs = EPOCHS;
        if (argc > 3)
        {
            epochs = atoi(argv[3]);
        }
        tireImagePaths = getImages(tiresPath);
        noTireImagePaths = getImages(nonTiresPath);
        tireImagesVectorSize = tireImagePaths.size();
        noTireImagesVectorSize = noTireImagePaths.size();
        totalSize = tireImagesVectorSize + noTireImagesVectorSize;
        magVectors = (float *)malloc(sizeof(float) * totalSize * X_DIM * Y_DIM);
        anglesVectors = (float *)malloc(sizeof(float) * totalSize * X_DIM * Y_DIM);
        features = (float *)malloc(sizeof(float) * totalSize * FEATURE_VECTOR_SIZE);

        labels = (unsigned char *)malloc(sizeof(unsigned char) * totalSize);

#pragma omp parallel num_threads(OMP_NUM_THREADS)
        {
            for (i = omp_get_thread_num(); i < tireImagesVectorSize; i += OMP_NUM_THREADS)
            {
                preprocessImage(string(tiresPath) + tireImagePaths[i],
                                magVectors, anglesVectors, i);
                labels[i] = 1;
            }

            for (i = omp_get_thread_num() + tireImagesVectorSize; i < totalSize; i += OMP_NUM_THREADS)
            {
                preprocessImage(string(nonTiresPath) + noTireImagePaths[i - tireImagesVectorSize],
                                magVectors, anglesVectors, i);
                labels[i] = 0;
            }
        }
        getFeatureVector(magVectors, anglesVectors, features, totalSize);

        trainLogRegression(epochs, totalSize, features, labels, weights);

        std::free(features);
        std::free(magVectors);
        std::free(anglesVectors);
        std::free(labels);
        cudaFree(Cj);

        ofstream weightsFile(WEIGHTS_FILE_PATH, ios::out);
        if (!weightsFile.is_open())
        {
            perror("Unable to save weights to file\n");
            return -1;
        }

        for (i = 0; i < FEATURE_VECTOR_SIZE + 1; i++)
        {
            snprintf(floatToStr, 20, "%f ", weights[i]);
            weightsFile << floatToStr;
        }

        weightsFile.close();

        return 0;
    }

    // If only a CLI parameter is provided, it expects it to be an image path
    // so it predicts if its a tire or not
    if (argc > 1)
    {
        if (loadWeights(weights))
        {
            cerr << "You must train model first, pass two folders: the "
                 << "first one containing tires images and the second one "
                 << "containing non-tires images" << endl;
            return -1;
        }
        magVectors = (float *)malloc(sizeof(float) * X_DIM * Y_DIM);
        anglesVectors = (float *)malloc(sizeof(float) * X_DIM * Y_DIM);
        features = (float *)malloc(sizeof(float) * FEATURE_VECTOR_SIZE);
        preprocessImage(string(argv[1]), magVectors, anglesVectors, 0);
        getFeatureVector(magVectors, anglesVectors, features, 1);
        drawFeatureVector(features);
        prediction = predict(features, weights);

        if (prediction > 0.5)
        {
            printf("This picture represents a tire, confidence: %.1f%%",
                   prediction * 100);
            cout << endl;
            return 0;
        }
        printf("This picture does not represent a tire, confidence: %.1f%%",
               (1 - prediction) * 100);
        cout << endl;
        return 0;
    }

    cerr << "You must pass either a path containing an image or two paths: "
         << "the first one containing tires images and the second one "
         << "containing non-tires images" << endl;
    return -1;
}
