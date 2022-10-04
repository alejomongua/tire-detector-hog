#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

#define EPSILON 1e-10

vector<string> getImages(const char* dirname)
{
    const unsigned int BATCH_SIZE = 1024;
    struct dirent* entry = NULL;
    string filename;
    DIR* dp = NULL;
    int stringCounter = 0;
    vector<string> output;

    dp = opendir(dirname);

    if (dp == NULL)
    {
        perror("Directorio incorrecto\n");
        return vector<string>();
    }

    while ((entry = readdir(dp)))
    {
        filename = string(entry->d_name);

        if (filename.find(".jpg") == -1)
            continue;

        output.push_back(filename);
    }

    closedir(dp);
    return output;
}

unsigned char calculateJ(float angle) {
    return (unsigned char)(angle / 20);
}

float calculateValueJ(float magnitude, float angle, float Cj) {
    return magnitude * (Cj - angle) / 20;
}

int getFeatureVector(string path, float* featureVector) {
    unsigned int i, j, k, l, m, tempI, tempJ, baseIndex;
    Mat img, resizedImg, gx, gy, mag, angle;
    Mat magnitudeValues, angleValues, histograms, output;
    float histogram[8][8][9], Cj[10], vJ, vJp1, singleAngle, singleMagnitude, norm;
    unsigned char valueJ;

    for (i = 0; i < 10; i++) {
        Cj[i] = 20 * (i + 0.5);
    }

    // Load images in an opencv matrix in gray scale
    img = imread(path, IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cerr << "Could not read the image: " << path << endl;
        return 1;
    }

    // Resize image
    resize(img, resizedImg, Size(64, 64));

    // Calculate gradients
    Sobel(resizedImg, gx, CV_32F, 1, 0, 1);
    Sobel(resizedImg, gy, CV_32F, 0, 1, 1);

    // Convert to polar
    cartToPolar(gx, gy, mag, angle, 1);

    // Process 8 x 8 blocks
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 8; j++) {

            tempI = i * 8;
            tempJ = j * 8;
            // Extract block
            magnitudeValues = mag(Range(tempI, tempI + 8), Range(tempJ, tempJ + 8));
            angleValues = angle(Range(tempI, tempI + 8), Range(tempJ, tempJ + 8));
            // Initialize histogram:
            for (k = 0; k < 9; k++) {
                histogram[i][j][k] = 0.0;
            }
            // Fill histogram
            for (k = 0; k < 8; k++) {
                for (l = 0; l < 8; l++) {
                    singleAngle = angleValues.at<float>(k, l);
                    singleMagnitude = magnitudeValues.at<float>(k, l);
                    if (singleAngle >= 180.0) {
                        singleAngle = singleAngle - 180;
                        singleMagnitude = -singleMagnitude;
                    }
                    valueJ = calculateJ(singleAngle);
                    vJ = calculateValueJ(singleMagnitude, singleAngle, Cj[valueJ]);
                    vJp1 = singleMagnitude - vJ;
                    histogram[i][j][valueJ] = vJ;
                    histogram[i][j][(valueJ) % 9] = vJp1;
                }
            }
        }
    }

    // Normalize
    for (i = 0; i < 7; i++) {
        for (j = 0; j < 7; j++) {
            // baseIndex needs to be calculated only once
            baseIndex = i * 252 + j * 36;
            for (m = 0; m < 9; m++) {
                // Square root of sum of squares
                norm = sqrt(
                    pow(histogram[i][j][m], 2) +
                    pow(histogram[i + 1][j][m], 2) +
                    pow(histogram[i][j + 1][m], 2) +
                    pow(histogram[i + 1][j + 1][m], 2));
                // EPSILON IS A SMALL NUMBER TO AVOID DIVISION BY 0
                // 36 * 7 = 252
                featureVector[baseIndex + m * 4] = histogram[i][j][m] / (norm + EPSILON);
                featureVector[baseIndex + m * 4 + 1] = histogram[i + 1][j][m] / (norm + EPSILON);
                featureVector[baseIndex + m * 4 + 2] = histogram[i][j + 1][m] / (norm + EPSILON);
                featureVector[baseIndex + m * 4 + 3] = histogram[i + 1][j + 1][m] / (norm + EPSILON);
            }
        }
    }
    return 0;
}

int main(int argc, const char** argv)
{
    const char* dirname = argc > 1
        ? argv[1]
        : "/root/sistemas_distribuidos/dataset/dataset1/llantas/";
    unsigned int index = 0, i, j, k;
    vector<string> imagePaths = getImages(dirname);
    float featureVector[7 * 7 * 36];
    unsigned int vectorSize = imagePaths.size();

    for (i = 0; i < vectorSize; i++) {
        getFeatureVector(string(dirname) + imagePaths[i], featureVector);
    }

    return 0;
}