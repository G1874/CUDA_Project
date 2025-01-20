#include <iostream>
#include <fstream>
#include <random>
#include <cmath>


typedef enum {
    RELU,
    SIGMOID,
    NO_ACTIVATION_FCN
} activationFcn_type;


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

void initialize_weights(float* weights, int size)
{
    std::mt19937 g(time(0));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for(int i = 0; i < size; i++)
    {
        weights[i] = dist(g);
    }
}

void initialize_biases(float* biases, int size)
{
    for(int i = 0; i < size; i++)
    {
        biases[i] = 0.0f;
    }
}


__device__ float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}


__device__ float reluDerivative(float x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}


__device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}


__device__ float sigmoidDerivative(float x)
{
    return sigmoid(x) * (1.0f - sigmoid(x));
}


__global__ void linearLayerForward(float* input, float* output, float* weights, float* biases,
                                   int inputSize, int outputSize, int batchSize, activationFcn_type activation)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int output_idx = batch_idx + threadIdx.x;
    int sample_idx = batch_idx * inputSize;

    float sum = 0.0f;
    for(int i=0; i<inputSize; i++)
    {
        sum += input[sample_idx + i] * weights[output_idx*inputSize + i];
    }
    sum += biases[output_idx];

    switch(activation)
    {
        case RELU:
            output[output_idx] = relu(sum);
            break;
        case SIGMOID:
            output[output_idx] = sigmoid(sum);
            break;
        case NO_ACTIVATION_FCN:
            output[output_idx] = sum;
            break;
    }
}


int main()
{
    const int input_size = 28 * 28;
    const int output_size = 10;
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

    //allocate host memory
    float* weights1 = new float[input_size * hidden_size];
    float* weights2 = new float[hidden_size * hidden_size];
    float* weights3 = new float[hidden_size * output_size];
    float* biases1 = new float[hidden_size];
    float* biases2 = new float[hidden_size];
    float* biases3 = new float[output_size];

    //initialize weights and biases
    initialize_weights(weights1, input_size * hidden_size);
    initialize_weights(weights2, hidden_size * hidden_size);
    initialize_weights(weights3, hidden_size * output_size);

    initialize_biases(biases1, hidden_size);
    initialize_biases(biases2, hidden_size);
    initialize_biases(biases3, output_size);

    //allocate device memory
    float *d_input, *d_output, *d_weights1, *d_weights2, *d_weights3, *d_biases1, *d_biases2, *d_biases3;

    cudaMalloc(&d_input, input_size * batch_size * sizeof(float));
    cudaMalloc(&d_output, input_size * batch_size * sizeof(float));
    cudaMalloc(&d_weights1, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_weights2, hidden_size * hidden_size * sizeof(float));
    cudaMalloc(&d_weights3, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_biases1, hidden_size * sizeof(float));
    cudaMalloc(&d_biases2, hidden_size * sizeof(float));
    cudaMalloc(&d_biases3, output_size * sizeof(float));


    //copy data to device
    cudaMemcpy(d_input, x_train, input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights1, weights1, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights2, weights2, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights3, weights3, hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases1, biases1, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases2, biases2, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases3, biases3, output_size * sizeof(float), cudaMemcpyHostToDevice);


    //forward pass
    for(int epoch = 0; epoch < 1; epoch++)
    {
        for(int i = 0; i < 60000 / batch_size; i++)
        {
        linearLayerForward<<<batch_size, hidden_size>>>(d_input, d_output, d_weights1, d_biases1, input_size, hidden_size, batch_size, RELU);
        cudaMemcpy(d_input,d_output, hidden_size * batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        linearLayerForward<<<batch_size, hidden_size>>>(d_input, d_output, d_weights2, d_biases2, hidden_size, hidden_size, batch_size, RELU);
        cudaMemcpy(d_input,d_output, hidden_size * batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        linearLayerForward<<<batch_size, output_size>>>(d_input, d_output, d_weights3, d_biases3, hidden_size, output_size, batch_size, NO_ACTIVATION_FCN);
        cudaMemcpy(d_input,d_output, hidden_size * batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }


    //free host memory
    delete[] x_train;
    delete[] y_train;
    delete[] x_test;
    delete[] y_test;
    delete[] weights1;
    delete[] weights2;
    delete[] weights3;
    delete[] biases1;
    delete[] biases2;
    delete[] biases3;

    //free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights1);
    cudaFree(d_weights2);
    cudaFree(d_weights3);
    cudaFree(d_biases1);
    cudaFree(d_biases2);
    cudaFree(d_biases3);
}