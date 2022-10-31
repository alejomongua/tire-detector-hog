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

#include "listDir.hpp"

using namespace std;
using namespace cv;

#define EPSILON 1e-10
#define PI 3.1416
#define FEATURE_VECTOR_SIZE 1764 // = 7 * 7 * 36
#define EULER 2.71828
#define EPOCHS 100
#define WEIGHTS_FILE_PATH "weights.data"
const float cosine[9] = {
    1,
    0.939692,
    0.766043,
    0.499998,
    0.173645,
    -0.173652,
    -0.500004,
    -0.766048,
    -0.939695 };
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
__constant__ float Cj[10];

__global__ void computeHistograms(float* angle, float* mag, float* hist, float* features)
{
    int i = blockIdx.x;
    int j = threadIdx.x;

    float histogram[9];

    // Initialize histogram:
    for (int k = 0; k < 9; k++)
    {
        histogram[k] = 0.0;
    }

    // Fill histogram
    for (int k = 0; k < 8; k++)
    {
        for (int l = 0; l < 8; l++)
        {
            unsigned int coordX = k + i * 8;
            unsigned int coordY = l + j * 8;
            unsigned int index = coordX * 64 + coordY;
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
            histogram[valueJ] += vJp1;
            // If valueJ is the last index, it should be 0 and change its sign
            if (valueJ == 8)
            {
                histogram[0] -= vJ;
            }
            else
            {
                histogram[(valueJ + 1) % 9] += vJ;
            }
        }
    }

    // Copy histogram to main GPU memory:
    for (int k = 0; k < 9; k++)
    {
        hist[i * 8 * 9 + j * 9 + k] = histogram[k];
    }

    __syncthreads();


    // Normalize
    if (i < 7 && j < 7)
    {
        // baseIndex needs to be calculated only once
        unsigned int baseIndex = i * 252 + j * 36;
        float powerSum = 0.0;
        for (int m = 0; m < 9; m++)
        {
            // Square root of sum of squares
            powerSum += pow(hist[i * 8 * 9 + j * 9 + m], 2) +
                pow(hist[(i + 1) * 8 * 9 + j * 9 + m], 2) +
                pow(hist[i * 8 * 9 + (j + 1) * 8 + m], 2) +
                pow(hist[(i + 1) * 8 * 9 + (j + 1) * 8 + m], 2);
        }
        float norm = sqrt(powerSum);

        for (int m = 0; m < 9; m++)
        {
            // EPSILON IS A SMALL NUMBER TO AVOID DIVISION BY 0
            // 36 * 7 = 252
            features[baseIndex + m * 4] =
                hist[i * 8 * 9 + j * 9 + m] / (norm + EPSILON);
            features[baseIndex + m * 4 + 1] =
                hist[(i + 1) * 8 * 9 + j * 9 + m] / (norm + EPSILON);
            features[baseIndex + m * 4 + 2] =
                hist[i * 8 * 9 + (j + 1) * 8 + m] / (norm + EPSILON);
            features[baseIndex + m * 4 + 3] =
                hist[(i + 1) * 8 * 9 + (j + 1) * 8 + m] / (norm + EPSILON);
        }
    }
}

int getFeatureVector(string path, float* h_features)
{
    Mat img, resizedImg, gx, gy, mag, angle;

    // Load images in an opencv matrix in gray scale
    img = imread(path, IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cerr << "Could not read the image: " << path << endl;
        return 1;
    }

    // Resize image
    // resize(img, resizedImg, Size(896, 896));
    // imshow("original", resizedImg);

    // Resize image
    resize(img, resizedImg, Size(64, 64));
    img.release();

    // Calculate gradients
    cv::Sobel(resizedImg, gx, CV_32F, 1, 0, 1);
    cv::Sobel(resizedImg, gy, CV_32F, 0, 1, 1);
    resizedImg.release();

    // Convert to polar
    cv::cartToPolar(gx, gy, mag, angle, 1);
    gx.release();
    gy.release();

    // Usar código de CUDA
    // Allocate memory
    float* d_angle = NULL;
    cudaError_t err = cudaMalloc((void**)&d_angle, 64 * 64 * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector angles (error: %s)!\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }

    float* d_mag = NULL;
    err = cudaMalloc((void**)&d_mag, 64 * 64 * sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector magnitudes (error: %s)!\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }

    float* d_histogram = NULL;
    err = cudaMalloc((void**)&d_histogram, 8 * 8 * 9 * sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error: %s)!\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }

    float* d_features = NULL;
    err = cudaMalloc((void**)&d_features, 8 * 8 * 9 * sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error: %s)!\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }

    // Pass parameters
    err = cudaMemcpy(d_angle, (float*)angle.data, 64 * 64 * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector angles from host to device (error: %s)!\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_mag, (float*)mag.data, 64 * 64 * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector magnitudes from host to device (error: %s)!\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
    angle.release();
    mag.release();

    // Launch kernel
    computeHistograms << <8, 8 >> > (d_angle, d_mag, d_histogram, d_features);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error: %s)!\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_features, d_features, FEATURE_VECTOR_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy features from device to host (error: %s)!\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_angle);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector angles (error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_mag);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector magnitudes (error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free device global memory
    err = cudaFree(d_histogram);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device histograms (error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free device global memory
    err = cudaFree(d_features);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device features (error: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

float predict(float* featureVector, float* weights)
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

void trainLogRegression(unsigned int epochs, unsigned int examples, float** features,
    unsigned char* labels, float* weights)
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
            prediction = predict(features[j], weights);
            error = labels[j] - prediction;
            slope = alpha * error;
            // cout << "Error=" << error << endl;
            weights[0] = weights[0] + slope;
            for (k = 1; k < FEATURE_VECTOR_SIZE + 1; k++)
            {
                weights[k] = weights[k] + slope * features[j][k - 1];
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
unsigned char loadWeights(float* weights)
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

void drawVector(Mat* img, float* featureVector, unsigned int baseIndex, unsigned char index)
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
void drawFeatureVector(float* featureVector)
{
    unsigned int i, j, k;
    Mat img, img1, img2, img3[4], img4, img5;

    for (i = 0; i < 7; i++)
    {
        for (j = 0; j < 7; j++)
        {
            for (k = 0; k < 4; k++)
            {
                drawVector(&img3[k], featureVector, i * 252 + j * 36, k);
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

int main(int argc, const char** argv)
{
    const char* nonTiresPath;
    const char* tiresPath;
    vector<string> tireImagePaths, noTireImagePaths;
    unsigned int i, epochs;
    unsigned int tireImagesVectorSize, noTireImagesVectorSize, totalSize;
    float featureVector[FEATURE_VECTOR_SIZE], weights[FEATURE_VECTOR_SIZE + 1];
    float prediction;
    float** features;
    unsigned char* labels;
    char floatToStr[20];
    float h_Cj[10];

    for (i = 0; i < 10; i++)
    {
        h_Cj[i] = 20.0 * i;
    }

    // copy Cj
    cudaError_t err = cudaMemcpyToSymbol(Cj, &h_Cj[0], 10 * sizeof(float), size_t(0), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy Cj to device (error: %s)!\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }

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
        features = (float**)malloc(sizeof(float*) * totalSize);

        labels = (unsigned char*)malloc(sizeof(unsigned char) * totalSize);

        for (i = 0; i < tireImagesVectorSize; i++)
        {
            features[i] = (float*)malloc(sizeof(float) * FEATURE_VECTOR_SIZE);
            getFeatureVector(string(tiresPath) + tireImagePaths[i],
                features[i]);
            labels[i] = 1;
        }

        for (i = tireImagesVectorSize; i < totalSize; i++)
        {
            features[i] = (float*)malloc(sizeof(float) * FEATURE_VECTOR_SIZE);
            getFeatureVector(string(nonTiresPath) + noTireImagePaths[i - tireImagesVectorSize],
                features[i]);
            labels[i] = 0;
        }

        trainLogRegression(epochs, totalSize, features, labels, weights);

        for (i = 0; i < totalSize; i++)
        {
            std::free(features[i]);
        }
        std::free(features);
        std::free(labels);

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
        getFeatureVector(string(argv[1]), featureVector);
        drawFeatureVector(featureVector);
        prediction = predict(featureVector, weights);

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