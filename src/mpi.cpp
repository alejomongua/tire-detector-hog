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
#include <mpi.h>

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
float Cj[10];

int getFeatureVector(string path, float* featureVector)
{
    Mat img, resizedImg, gx, gy, mag, angle;
    float histogram[8][8][9];

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
    Sobel(resizedImg, gx, CV_32F, 1, 0, 1);
    Sobel(resizedImg, gy, CV_32F, 0, 1, 1);
    resizedImg.release();

    // Convert to polar
    cartToPolar(gx, gy, mag, angle, 1);
    gx.release();
    gy.release();

    // Process 8 x 8 blocks
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            // Initialize histogram:
            for (int k = 0; k < 9; k++)
            {
                histogram[i][j][k] = 0.0;
            }
            // Fill histogram
            for (int k = 0; k < 8; k++)
            {
                for (int l = 0; l < 8; l++)
                {
                    unsigned int coordX = k + i * 8;
                    unsigned int coordY = l + j * 8;
                    float singleAngle = angle.at<float>(coordX, coordY);
                    float singleMagnitude = mag.at<float>(coordX, coordY);
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
                        histogram[i][j][(valueJ + 1) % 9] += vJ;
                    }
                }
            }
        }
    }

    angle.release();
    mag.release();

    // Normalize
    for (int i = 0; i < 7; i++)
    {
        for (int j = 0; j < 7; j++)
        {
            // baseIndex needs to be calculated only once
            unsigned int baseIndex = i * 252 + j * 36;
            float powerSum = 0.0;
            for (int m = 0; m < 9; m++)
            {
                // Square root of sum of squares
                powerSum += pow(histogram[i][j][m], 2) +
                    pow(histogram[i + 1][j][m], 2) +
                    pow(histogram[i][j + 1][m], 2) +
                    pow(histogram[i + 1][j + 1][m], 2);
            }
            float norm = sqrt(powerSum);

            for (int m = 0; m < 9; m++)
            {
                // EPSILON IS A SMALL NUMBER TO AVOID DIVISION BY 0
                // 36 * 7 = 252
                featureVector[baseIndex + m * 4] =
                    histogram[i][j][m] / (norm + EPSILON);
                featureVector[baseIndex + m * 4 + 1] =
                    histogram[i + 1][j][m] / (norm + EPSILON);
                featureVector[baseIndex + m * 4 + 2] =
                    histogram[i][j + 1][m] / (norm + EPSILON);
                featureVector[baseIndex + m * 4 + 3] =
                    histogram[i + 1][j + 1][m] / (norm + EPSILON);
            }
        }
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

    // cout << "costo = [";
    for (i = 0; i < epochs; i++)
    {
        if (!(i % 100))
        {
            // printf("Epoch %d/%d...\n", i, epochs);
        }

        float costo = 0;
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
            costo += cost(labels[j], prediction);
        }
        // cout << costo / examples << ", ";
    }
    // cout << "]" << endl;
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

int main(int argc, char* argv[])
{
    const char* nonTiresPath;
    const char* tiresPath;
    vector<string> tireImagePaths, noTireImagePaths;
    unsigned int i = 0, epochs;
    unsigned int tireImagesVectorSize, noTireImagesVectorSize, totalSize;
    float featureVector[FEATURE_VECTOR_SIZE], weights[FEATURE_VECTOR_SIZE + 1];
    float prediction;
    float** features;
    unsigned char* labels;
    char floatToStr[20];


    for (i = 0; i < 10; i++)
    {
        Cj[i] = 20.0 * i;
    }

    // If two paths are passed, it should train
    // the first path is the folder with tires, and the second path is the
    // non-tires folder
    if (argc > 2)
    {
        int myid, numprocs;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
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
        for (int i = myid; i < tireImagesVectorSize; i += numprocs)
        {
            features[i] = (float*)malloc(sizeof(float) * FEATURE_VECTOR_SIZE);
            getFeatureVector(string(tiresPath) + tireImagePaths[i],
                features[i]);
        }

        for (int i = tireImagesVectorSize + myid; i < totalSize; i += numprocs)
        {
            features[i] = (float*)malloc(sizeof(float) * FEATURE_VECTOR_SIZE);
            getFeatureVector(string(nonTiresPath) + noTireImagePaths[i - tireImagesVectorSize],
                features[i]);
        }

        /* Ahora transmitimos la informaci??n desde los otros procesos al master */
        for (int i = 0; i < totalSize; i++)
        {
            MPI_Status status;
            // El proceso 0 debe ejecutarse en todas las im??genes para recibir la informaci??n
            // Los dem??s procesos se saltan esta ejecuci??n cuando no sea su turno
            if (myid != 0 && i % numprocs != myid) continue;

            // cout << "Proceso " << myid << "; imagen " << i << endl;
            if (myid == 0) {
                // Ponga los labels en cada imagen
                labels[i] = i < tireImagesVectorSize ? 1 : 0;
                // Si no tiene la informaci??n, traigala desde el proceso correspondiente
                if (i % numprocs != 0) {
                    features[i] = (float*)malloc(sizeof(float) * FEATURE_VECTOR_SIZE);
                    MPI_Recv(features[i], FEATURE_VECTOR_SIZE, MPI_FLOAT, i % numprocs, 1, MPI_COMM_WORLD, &status);
                }
            }
            else {
                MPI_Send(features[i], FEATURE_VECTOR_SIZE, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            }
        }

        if (myid == 0) {
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
                MPI_Finalize();
                return -1;
            }

            for (i = 0; i < FEATURE_VECTOR_SIZE + 1; i++)
            {
                snprintf(floatToStr, 20, "%f ", weights[i]);
                weightsFile << floatToStr;
            }

            weightsFile.close();
        }
        else {
            // cout << "Proceso " << myid << ", hasta aqu?? todo bien" << endl;
        }
        return MPI_Finalize();
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
