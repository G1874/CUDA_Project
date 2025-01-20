#include <iostream>
#include <cmath>

typedef enum {
    RELU,
    SIGMOID,
    SOFTMAX
} activationFunction_type;

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


__device__ relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

__device__ reluDerivative(float x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}

__device__ sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ sigmoidDerivative(float x)
{
    return sigmoid(x) * (1.0f - sigmoid(x));
}

__global__ void linearLayerForward(float* input, float* output, float* weights, float* biases,
                                   int inputSize, int outputSize, int batchSize, activationFcn_type activation)
{
    int idx = threadIdx.x;

    float sum = 0.0f;
    for(int i=0; i<inputSize, )
}


int main()
{
    const int hidden_size = 128;
    const int batch_size = 64;
    const int num_epochs = 10;
    const float learning_rate = 0.001f;

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

    
}