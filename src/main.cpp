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
#include <fstream>

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
    -0.939695
};
const float sine[9] = {
    0,
    0.342021,
    0.642789,
    0.866027,
    0.984808,
    0.984807,
    0.866023,
    0.642783,
    0.342014,
};

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
    float histogram[8][8][9], Cj[10], vJ, vJp1;
    float singleAngle, singleMagnitude, norm;
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
    resize(img, resizedImg, Size(896, 896));
    imshow("original", resizedImg);
    waitKey(0);

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
            magnitudeValues = mag(Range(tempI, tempI + 8),
                Range(tempJ, tempJ + 8));
            angleValues = angle(Range(tempI, tempI + 8),
                Range(tempJ, tempJ + 8));
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
                    vJ = calculateValueJ(singleMagnitude, singleAngle,
                        Cj[valueJ]);
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

float predict(float* featureVector, float* weights) {
    float sum = weights[0];
    unsigned int i;

    for (i = 0; i < FEATURE_VECTOR_SIZE; i++) {
        sum += weights[i + 1] * featureVector[i];
    }

    return 1 / (1 + pow(EULER, -sum));
}

void shuffle(unsigned char* labels, float** examples, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            float* t;
            t = examples[j];
            examples[j] = examples[i];
            examples[i] = t;
            unsigned char w = labels[j];
            labels[j] = labels[i];
            labels[i] = w;
        }
    }
}

float cost(unsigned char label, float prediction) {
    float cost1 = label * log(prediction);
    float cost2 = (1 - label) * log(1 - prediction);
    return -cost1 - cost2;
}

void trainLogRegression(unsigned int epochs, unsigned int examples, float** features,
    unsigned char* labels, float* weights) {
    unsigned int i, j, k;
    float prediction, error;
    float alpha = 0.001; // learning rate
    for (i = 0; i < FEATURE_VECTOR_SIZE + 1; i++) {
        weights[i] = 0;
    }

    for (i = 0; i < epochs; i++) {
        shuffle(labels, features, examples);
        if (!(i % 100)) {
            printf("Epoch %d/%d...\n", i, epochs);
        }

        float costo = 0;
        for (j = 0; j < examples; j++) {
            prediction = predict(features[j], weights);
            error = labels[j] - prediction;
            // cout << "Error=" << error << endl;
            weights[0] = weights[0] + alpha * error;
            for (k = 0; k < FEATURE_VECTOR_SIZE; k++) {
                weights[k + 1] = weights[k + 1] + alpha * error * features[j][k];
            }
            costo += cost(labels[j], prediction);
        }
        cout << "Costo=" << costo / examples << endl;
    }
}

/*
 * This function loads weights from weights file
 * it returns 1 if there are no weights or 0 if
 * weights are loaded
 */
unsigned char loadWeights(float* weights) {
    char floatToStr[20];
    unsigned int i = 0, j = 0;
    ifstream weightsInFile(WEIGHTS_FILE_PATH);

    if (!weightsInFile.good()) {
        return 1;
    }
    std::vector<int> load;
    char temp;

    while (weightsInFile.read(&temp, 1)) {
        floatToStr[i++] = temp;
        if (temp == ' ') {
            floatToStr[i] = 0;
            weights[j++] = atof(floatToStr);
            i = 0;
        }
    }

    weightsInFile.close();
    return 0;
}

void drawFeatureVector(float* featureVector) {
    unsigned int i, j, k, x, y;
    const int width = 64;
    Mat img, img1, img2;

    for (i = 0; i < 14; i++){
        for (j = 0; j < 14; j++) {
            if (!j){
                img = Mat::zeros(width, width, CV_32F);
                for (k = 0; k < 9; k++) {
                    x = (unsigned int)(featureVector[i * 126 + j * 9 + k] * cosine[k] * width + width / 2);
                    y = (unsigned int)(featureVector[i * 126 + j * 9 + k] * sine[k] * width + width / 2);
                    line(
                        img, 
                        Point( width / 2, width / 2 ),
                        Point( x, y ),
                        Scalar(255),
                        1,
                        LINE_8
                    );
                }
            } else {
                img1 = Mat::zeros(width, width, CV_32F);
                for (k = 0; k < 9; k++) {
                    x = (unsigned int)(featureVector[i * 126 + j * 9 + k] * cosine[k] * width + width / 2);
                    y = (unsigned int)(featureVector[i * 126 + j * 9 + k] * sine[k] * width + width / 2);
                    line(
                        img1, 
                        Point( width / 2, width / 2 ),
                        Point( x, y ),
                        Scalar(255),
                        1,
                        LINE_8
                    );
                }
                hconcat(img, img1, img);
            }
        }
        if (img2.empty()) {
            img2 = img.clone();
        } else {
            vconcat(img2, img, img2);
        }
    }

    imshow("Display window", img2);
    waitKey(0); // Wait for a keystroke in the window
}

int main(int argc, const char** argv)
{
    const char* nonTiresPath;
    const char* tiresPath;
    vector<string> tireImagePaths, noTireImagePaths;
    unsigned int i = 0, j = 0, k, epochs;
    unsigned int tireImagesVectorSize, noTireImagesVectorSize;
    float featureVector[FEATURE_VECTOR_SIZE], weights[FEATURE_VECTOR_SIZE + 1];
    float prediction;
    float** features;
    unsigned char* labels;
    char floatToStr[20];

    // If two paths are passed, it should train
    // the first path is the folder with tires, and the second path is the
    // non-tires folder
    if (argc > 2) {
        tiresPath = argv[1];
        nonTiresPath = argv[2];
        epochs = EPOCHS;
        if (argc > 3) {
            epochs = atoi(argv[3]);
        }
        tireImagePaths = getImages(tiresPath);
        noTireImagePaths = getImages(nonTiresPath);
        tireImagesVectorSize = tireImagePaths.size();
        noTireImagesVectorSize = noTireImagePaths.size();
        features = (float**)malloc(sizeof(float) * FEATURE_VECTOR_SIZE
            * (tireImagesVectorSize + noTireImagesVectorSize));

        labels = (unsigned char*)malloc(sizeof(unsigned char)
            * (tireImagesVectorSize + noTireImagesVectorSize));

        for (i = 0; i < tireImagesVectorSize; i++) {
            getFeatureVector(string(tiresPath) + tireImagePaths[i],
                featureVector);
            features[i] = featureVector;
            labels[i] = 1;
        }

        for (i = 0; i < noTireImagesVectorSize; i++) {
            getFeatureVector(string(nonTiresPath) + noTireImagePaths[i],
                featureVector);
            features[i + tireImagesVectorSize] = featureVector;
            labels[i + tireImagesVectorSize] = 0;
        }

        trainLogRegression(epochs, tireImagesVectorSize + noTireImagesVectorSize,
            features, labels, weights);

        ofstream weightsFile(WEIGHTS_FILE_PATH, ios::out);
        if (!weightsFile.is_open())
        {
            perror("Unable to save weights to file\n");
            return -1;
        }

        for (i = 0; i < FEATURE_VECTOR_SIZE + 1; i++) {
            snprintf(floatToStr, 20, "%f ", weights[i]);
            weightsFile << floatToStr;
        }

        weightsFile.close();


        return 0;
    }

    // If only a CLI parameter is provided, it expects it to be an image path
    // so it predicts if its a tire or not
    if (argc > 1) {
        if (loadWeights(weights)) {
            cerr << "You must train model first, pass two folders: the " <<
                "first one containing tires images and the second one " <<
                "containing non-tires images" << endl;
            return -1;
        }
        getFeatureVector(string(argv[1]), featureVector);
        drawFeatureVector(featureVector);
        prediction = predict(featureVector, weights);

        if (prediction > 0.5) {
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

    cerr << "You must pass either a path containing an image or two paths: " <<
        "the first one containing tires images and the second one " <<
        "containing non-tires images" << endl;
    return -1;

}

