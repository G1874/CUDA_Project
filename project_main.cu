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


__global__ float relu(float* input, float* output)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int idx = batch_idx * threadIdx.x;

    output[idx] = (x > 0.0f) ? x : 0.0f;
}


__global__ float reluDerivative(float* input, float* output)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int idx = batch_idx * threadIdx.x;

    output[idx] = (x > 0.0f) ? 1.0f : 0.0f;
}


__global__ float sigmoid(float* input, float* output)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int idx = batch_idx * threadIdx.x;

    output[idx] = 1.0f / (1.0f + expf(-x));
}


__global__ float sigmoidDerivative(float* input, float* output)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int idx = batch_idx * threadIdx.x;

    output[idx] = sigmoid(x) * (1.0f - sigmoid(x));
}


__global__ void crossEntropyLoss(float* predictions, float* labels, float* loss)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int idx = batch_idx * threadIdx.x;

    float l = -labels[idx] * logf(predictions[idx]);

    atomicAdd(loss[batch_idx], l);
}


__global__ void crossEntropyLossDerivative(float* predictions, float* labels, float* loss)
{
    // TODO:
}


__global__ void linearLayerForward(float* input, float* output, float* weights, float* biases,
                                   int inputSize, int outputSize, int batchSize)
{
    int batch_idx = blockIdx.x * blockDim.x;
    int output_idx = batch_idx + threadIdx.x;
    int sample_idx = batch_idx * inputSize;

    float sum = 0.0f;
    for(int i=0; i<inputSize; i++)
    {
        sum += input[sample_idx + i] * weights[output_idx*inputSize + i];
    }
    
    output[output_idx] = sum + biases[output_idx];
}


__global__ void linearLayerBackward()
{
    
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

    //intialize layers
    layer_type layer1, layer2, layer3;

    //layer 1
    layer1.weights = new float[input_size * hidden_size];
    layer1.biases = new float[hidden_size];
    layer1.inputs = new float[input_size * batch_size];

    //layer 2
    layer2.weights = new float[hidden_size * hidden_size];
    layer2.biases = new float[hidden_size];

    //layer 3
    layer3.weights = new float[hidden_size * output_size];
    layer3.biases = new float[output_size];
    layer3.outputs = new float[output_size * batch_size];


    //initialize weights and biases
    initialize_weights(layer1.weights, input_size * hidden_size);
    initialize_weights(layer2.weights, hidden_size * hidden_size);
    initialize_weights(layer3.weights, hidden_size * output_size);

    initialize_biases(layer1.biases, hidden_size);
    initialize_biases(layer2.biases, hidden_size);
    initialize_biases(layer3.biases, output_size);

    //allocate device memory
    layer_type d_layer1, d_layer2, d_layer3;

    cudaMalloc(&d_layer1.weights, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_layer1.biases, hidden_size * sizeof(float));
    cudaMalloc(&d_layer1.inputs, input_size * batch_size * sizeof(float));
    cudaMalloc(&d_layer1.outputs, hidden_size * batch_size * sizeof(float));

    cudaMalloc(&d_layer2.weights, hidden_size * hidden_size * sizeof(float));
    cudaMalloc(&d_layer2.biases, hidden_size * sizeof(float));
    cudaMalloc(&d_layer2.inputs, hidden_size * batch_size * sizeof(float));
    cudaMalloc(&d_layer2.outputs, hidden_size * batch_size * sizeof(float));

    cudaMalloc(&d_layer3.weights, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_layer3.biases, output_size * sizeof(float));
    cudaMalloc(&d_layer3.inputs, hidden_size * batch_size * sizeof(float));
    cudaMalloc(&d_layer3.outputs, output_size * batch_size * sizeof(float));


    //copy data to device
    cudaMemcpy(d_layer1.inputs, x_train, input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer1.weights, layer1.weights, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer1.biases, layer1.biases, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer2.weights, layer2.weights, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer2.biases, layer2.biases, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer3.weights, layer3.weights, hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer3.biases, layer3.biases, output_size * sizeof(float), cudaMemcpyHostToDevice);



    //forward pass
    for(int epoch = 0; epoch < 1; epoch++)
    {
        for(int i = 0; i < 60000 / batch_size; i++)
        {
        linearLayerForward<<<batch_size, hidden_size>>>(d_layer1.inputs, d_layer1.outputs, d_layer1.weights, d_layer1.biases, 
                                                        input_size, hidden_size, batch_size);
        cudaMemcpy(d_layer2.inputs,d_layer1.outputs, hidden_size * batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        relu<<<batch_size, hidden_size>>>(d_layer2.inputs, d_layer2.inputs);

        linearLayerForward<<<batch_size, hidden_size>>>(d_layer2.inputs, d_layer2.outputs, d_layer2.weights, d_layer2.biases,
                                                        hidden_size, hidden_size, batch_size);
        cudaMemcpy(d_layer3.inputs, d_layer2.outputs, hidden_size * batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
        relu<<<batch_size, hidden_size>>>(d_layer3.inputs, d_layer3.inputs);

        linearLayerForward<<<batch_size, output_size>>>(d_layer3.inputs, d_layer3.outputs, d_layer3.weights, d_layer3.biases,
                                                        hidden_size, output_size, batch_size);
        }
    }

    //free host memory
    delete[] x_train;
    delete[] y_train;
    delete[] x_test;
    delete[] y_test;

    //free device memory
    cudaFree(d_layer1.weights);
    cudaFree(d_layer1.biases);
    cudaFree(d_layer1.inputs);
    cudaFree(d_layer1.outputs);

    cudaFree(d_layer2.weights);
    cudaFree(d_layer2.biases);
    cudaFree(d_layer2.inputs);
    cudaFree(d_layer2.outputs);

    cudaFree(d_layer3.weights);
    cudaFree(d_layer3.biases);
    cudaFree(d_layer3.inputs);
    cudaFree(d_layer3.outputs);
}