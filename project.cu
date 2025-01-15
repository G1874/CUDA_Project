#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>

float Number_size = 28 * 28;

float* load_data(const char* filename, int size)
{
    std::ifstream file(filename);
    if(!file.is_open())
    {
        std::cerr << "Error: file not found" << std::endl;
        return NULL;
    }

    float* data = new float[size];
    for(int i = 0; i < size; i++)
    {
        file >> data[i];
        if(file.peek() == ',')
            file.ignore();
    }
    file.close();

    return data;
}


int main() 
{
    float* x_train;
    float* y_train;
    float* x_test;
    float* y_test;

    x_train = load_data("dataset/x_train.txt", 60000 * Number_size);
    y_test = load_data("dataset/y_train.txt", 60000);
    x_test = load_data("dataset/x_test.txt", 10000 * Number_size);
    y_test = load_data("dataset/y_test.txt", 10000);


    if(x_train)
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
