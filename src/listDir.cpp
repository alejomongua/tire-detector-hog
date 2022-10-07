#include <string>
#include <vector>
#include <iostream>
#ifdef __unix__
#include <dirent.h>
#else
#include <windows.h>
#endif

using namespace std;

vector<string> getImages(const char* dirname)
{
    struct dirent* entry = NULL;
    string filename;
    int stringCounter = 0;
    vector<string> output;

#ifdef __unix__
    DIR* dp = NULL;
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
#else
    string pattern(dirname);
    pattern.append("\\*");
    WIN32_FIND_DATA data;
    HANDLE hFind;
    if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
        do {
            filename = string(data.cFileName);
            if (filename.find(".jpg") == -1)
                continue;
            output.push_back(filename);
        } while (FindNextFile(hFind, &data) != 0);
        FindClose(hFind);
    }
#endif
    return output;
}