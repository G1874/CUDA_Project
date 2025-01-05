#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>

void load_data(const char* filename, float*& data, int rows, int cols)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: file not found" << std::endl;
        return;
    }

    data = new float[rows * cols];
    for (int i = 0; i < rows * cols; i++)
    {
        file >> data[i];
        if (file.peek() == ',')
            file.ignore();
    }

    file.close();
}


int main() 
{
    float* x_train = nullptr;

    load_data("dataset/y_test.txt", x_train, 100, 1);

    if (x_train)
    {
        printf("First 100 elements of x_test:\n");
        for (int i = 0; i < 100; i++)
        {
            printf("%f ", x_train[i]);
        }
        printf("\n");

        delete[] x_train;
    }
    else
    {
        printf("Failed to load data\n");
    }

    return 0;
}
