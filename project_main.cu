#include <iostream>
#include <fstream>
#include <random>
#include <cmath>


typedef struct {
    float* weights;
    float* biases;
    float* inputs;
    float* outputs;
} layer_type;


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


__global__ void relu(float* input, float* output)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int idx = batch_idx * threadIdx.x;

    output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
}


__global__ void reluDerivative(float* input, float* output)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int idx = batch_idx * threadIdx.x;

    output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
}


__global__ void softmax(float* input, float* output)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int idx = batch_idx * threadIdx.x;

    extern __shared__ float sum[];

    atomicAdd(sum[batch_idx], expf(input[idx]));
    __syncthreads();

    output[idx] = expf(input[idx]) / sum[batch_idx];
}


__global__ void crossEntropyLoss(float* predictions, float* labels, float* loss)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int idx = batch_idx * threadIdx.x;

    float l = -labels[idx] * logf(predictions[idx]);

    atomicAdd(loss[batch_idx], l);
}


__device__ void crossEntropyLossDerivative(float* predictions, float* labels, float* output)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int idx = batch_idx * threadIdx.x;

    output[idx] = predictions[idx] - labels[idx];
}


__global__ void linearLayerForward(float* input, float* output, float* weights, float* biases,
                                   int inputSize, int outputSize, int batchSize)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int neuron_idx = threadIdx.x;
    int output_idx = batch_idx + neuron_idx;
    int sample_idx = batch_idx * inputSize;

    float sum = 0.0f;
    for(int i=0; i<inputSize; i++)
    {
        sum += input[sample_idx + i] * weights[neuron_idx*inputSize + i];
    }
    
    output[output_idx] = sum + biases[neuron_idx];
}


__global__ void linearLayerBackward(float* error_grad, float* inputs, float* input_grad,
                                    float* weights_grad, float* biases_grad, int batchSize)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int output_idx = batch_idx * threadIdx.x;
    int sample_idx = batch_idx * inputSize;

    for(int i=0; i<inputSize; i++)
    {
        weights_grad[output_idx*inputSize + i] = error_grad[output_idx] + inputs[sample_idx + i];
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
    float* weights = new float[input_size * hidden_size + hidden_size * hidden_size + hidden_size * output_size];
    float* biases = new float[hidden_size + hidden_size + output_size];

    //initialize weights and biases
    std::mt19937 g(time(0));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for(int i = 0; i < input_size * hidden_size + hidden_size * hidden_size + hidden_size * output_size; i++)
    {
        weights[i] = dist(g);
    }

    for(int i = 0; i < hidden_size + hidden_size + output_size; i++)
    {
        biases[i] = 0.0f;
    }

    //allocate device memory
    float *d_input, *d_hidden, *d_output, *d_weights, *d_biases;

    cudaMalloc(&d_input, input_size * batch_size * sizeof(float));
    cudaMalloc(&d_hidden, hidden_size * batch_size * sizeof(float));
    cudaMalloc(&d_output, output_size * batch_size * sizeof(float));
    cudaMalloc(&d_weights, input_size * hidden_size + hidden_size * hidden_size + hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_biases, hidden_size + hidden_size + output_size * sizeof(float));

    //copy data to device
    cudaMemcpy(d_input, x_train, input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, input_size * hidden_size + hidden_size * hidden_size + hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, hidden_size + hidden_size + output_size * sizeof(float), cudaMemcpyHostToDevice);

    //free host memory
    delete[] x_train;
    delete[] y_train;
    delete[] x_test;
    delete[] y_test;
    delete[] weights;
    delete[] biases;

    //free device memory
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
}