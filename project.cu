#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>

const int input_size = 28 * 28;
const int output_size = 10;


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

void normalize_data(float* data, int size)
{
    for(int i = 0; i < size; i++)
    {
        data[i] = data[i] / 255.0;
    }
}


int main() 
{
    const int hidden_size = 128;
    const int batch_size = 64;

    float* x_train;
    float* y_train;
    float* x_test;
    float* y_test;

    x_train = load_data("dataset/x_train.txt", 60000 * input_size);
    y_train = load_data("dataset/y_train.txt", 60000);
    x_test = load_data("dataset/x_test.txt", 10000 * input_size);
    y_test = load_data("dataset/y_test.txt", 10000);

    normalize_data(x_train, 60000 * input_size);
    normalize_data(x_test, 10000 * input_size);

    //allocate device memory
    float* d_input;
    float* d_hidden;
    float* d_output;
    float* d_weights1;
    float* d_bias1;
    float* d_weights2;
    float* d_bias2;

    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_hidden, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_weights1, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_bias1, hidden_size * sizeof(float));
    cudaMalloc(&d_weights2, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_bias2, output_size * sizeof(float));


    return 0;
}
