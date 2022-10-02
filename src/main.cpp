#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <string>

using namespace std;

string* getImages(const char* dirname)
{
    const unsigned int BATCH_SIZE = 32;
    struct dirent* entry = NULL;
    string filename;
    DIR* dp = NULL;
    int stringCounter = 0;
    string* output;

    dp = opendir(dirname);

    if (dp == NULL)
    {
        perror("Directorio incorrecto\n");
        return NULL;
    }

    output = (string*)malloc(sizeof(string) * BATCH_SIZE);
    if (output == NULL) {
        perror("Hubo un error aquí\n");
        return NULL;
    }

    while ((entry = readdir(dp)))
    {
        filename = string(entry->d_name);

        if (filename.find(".jpg") == -1)
            continue;

        if (stringCounter && !(stringCounter % BATCH_SIZE)) {
            output = (string*)realloc(output, sizeof(string) * BATCH_SIZE * (stringCounter / BATCH_SIZE + 1));
            if (output == NULL) {
                perror("Hubo un error aquí\n");
                return NULL;
            }
        }
        output[stringCounter++] = filename;
    }

    closedir(dp);
    return output;
}

int main(int argc, const char** argv)
{
    const char* dirname = argc > 1 ? argv[1] : ".";
    unsigned int i = 1;

    string* imagePaths = getImages(dirname);

    if (imagePaths == NULL)
    {
        perror("Hubo un error al leer las imágenes\n");
        return -1;
    }

    while (imagePaths[i] != "") {
        cout << imagePaths[i++] << endl;
    }

    free(imagePaths);

    return 0;
}