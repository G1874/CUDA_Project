#include <iostream>
#include <cudnn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
#include <vector>

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

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

    //allocate device memory
    float *d_input, *d_hidden, *d_output, *d_weights1, *d_bias1, *d_weights2, *d_bias2;

    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_hidden, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_weights1, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_bias1, hidden_size * sizeof(float));
    cudaMalloc(&d_weights2, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_bias2, output_size * sizeof(float));

    //initialize weights and biases
    std::vector<float> h_weights1(input_size * hidden_size, 0.1f);
    std::vector<float> h_bias1(hidden_size, 0.1f);
    std::vector<float> h_weights2(hidden_size * output_size, 0.1f);
    std::vector<float> h_bias2(output_size, 0.1f);

    cudaMemcpy(d_weights1, h_weights1.data(), input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias1, h_bias1.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights2, h_weights2.data(), hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias2, h_bias2.data(), output_size * sizeof(float), cudaMemcpyHostToDevice);

    //initialize cudnn
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    //initialize cudnn tensor descriptors
    cudnnTensorDescriptor_t input_desc, hidden_desc, output_desc, bias_desc1, bias_desc2;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&hidden_desc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc1));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc2));
    
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, input_size, 1, 1));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(hidden_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, hidden_size, 1, 1));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, output_size, 1, 1));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(bias_desc1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, hidden_size, 1, 1));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(bias_desc2, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_size, 1, 1));

    //create activation descriptors
    cudnnActivationDescriptor_t activation_desc;
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc));
    CUDNN_CALL(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

    float alpha = 1.0f;
    float beta = 0.0f;

    //training loop
    for(int epoch = 0; epoch < num_epochs; epoch++)
    {
        float epoch_loss = 0.0f;
        int num_batches = 60000 / batch_size;

        for(int batch = 0; batch < num_batches; batch++)
        {
            //load data
            cudaMemcpy(d_input, x_train + batch * batch_size * input_size, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
                
            //Forward pass: input -> hidden
            CUDNN_CALL(cudnnAddTensor(cudnn, &alpha, bias_desc1, d_bias1, &beta, hidden_desc, d_hidden));
            CUDNN_CALL(cudnnActivationForward(cudnn, activation_desc, &alpha, hidden_desc, d_hidden,
                                            &beta, hidden_desc, d_hidden));

            // // Forward pass: hidden -> output
            CUDNN_CALL(cudnnAddTensor(cudnn, &alpha, bias_desc2, d_bias2, &beta, output_desc, d_output));
            CUDNN_CALL(cudnnActivationForward(cudnn, activation_desc, &alpha, output_desc, d_output,
                                            &beta, output_desc, d_output));

            // Compute loss (Mean Squared Error)
            std::vector<float> h_output(batch_size * output_size);
            cudaMemcpy(h_output.data(), d_output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

            std::vector<float> h_loss(batch_size * output_size);
            for (int i = 0; i < batch_size * output_size; ++i) 
            {
                h_loss[i] = h_output[i] - static_cast<float>(y_train[batch * batch_size + i / output_size]);
                epoch_loss += 0.5f * h_loss[i] * h_loss[i];
            }

            cudaMemcpy(d_output, h_loss.data(), batch_size * output_size * sizeof(float), cudaMemcpyHostToDevice);

            // // Backpropagation: output -> hidden
            // CUDNN_CALL(cudnnActivationBackward(cudnn, activation_desc, &alpha, output_desc, d_output,
            //                                    output_desc, d_output, hidden_desc, d_hidden,
            //                                    &beta, hidden_desc, d_hidden));



            // // Update weights2 and bias2
            // for(int i = 0; i < hidden_size * output_size; ++i) 
            // {
            //     h_weights2[i] -= learning_rate * h_loss[i % output_size];
            // }
            // for(int i = 0; i < output_size; ++i) 
            // {
            //     h_bias2[i] -= learning_rate * h_loss[i];
            // }

            // cudaMemcpy(d_weights2, h_weights2.data(), hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
            // // cudaMemcpy(d_bias2, h_bias2.data(), output_size * sizeof(float), cudaMemcpyHostToDevice);

            // // Backpropagation: hidden -> input
            // CUDNN_CALL(cudnnActivationBackward(cudnn, activation_desc, &alpha, hidden_desc, d_hidden,
            //                                    hidden_desc, d_hidden, input_desc, d_input,
            //                                    &beta, input_desc, d_input));
            
            // // Update weights1 and bias1
            // for(int i = 0; i < input_size * hidden_size; ++i) 
            // {
            //     h_weights1[i] -= learning_rate * h_loss[i % hidden_size];
            // }
            // for(int i = 0; i < hidden_size; ++i) 
            // {
            //     h_bias1[i] -= learning_rate * h_loss[i];
            // }

            // cudaMemcpy(d_weights1, h_weights1.data(), input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
            // cudaMemcpy(d_bias1, h_bias1.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice);

        }

        // //Evaluate model every epoch
        // cudaMemcpy(d_input, x_test, 10000 * input_size * sizeof(float), cudaMemcpyHostToDevice);
        // CUDNN_CALL(cudnnAddTensor(cudnn, &alpha, input_desc, d_input, &beta, hidden_desc, d_bias1));
        // CUDNN_CALL(cudnnActivationForward(cudnn, activation_desc, &alpha, hidden_desc, d_hidden,
        //                                   &beta, hidden_desc, d_hidden));
        // CUDNN_CALL(cudnnAddTensor(cudnn, &alpha, hidden_desc, d_hidden, &beta, output_desc, d_bias2));
    }

    //clean up
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(hidden_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bias_desc1));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bias_desc2));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc));
    CUDNN_CALL(cudnnDestroy(cudnn));

    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_weights1);
    cudaFree(d_bias1);
    cudaFree(d_weights2);
    cudaFree(d_bias2);

    return 0;
}