#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Network architecture
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define BATCH_SIZE 64
#define LEARNING_RATE 0.01f
#define EPOCHS 10

// Activation functions
__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// Softmax (for output layer)
__global__ void softmax_kernel(float* input, float* output, int batch_size, int size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        float* in_row = input + batch_idx * size;
        float* out_row = output + batch_idx * size;
        
        // Find max for numerical stability
        float max_val = in_row[0];
        for (int i = 1; i < size; i++) {
            max_val = fmaxf(max_val, in_row[i]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            out_row[i] = expf(in_row[i] - max_val);
            sum += out_row[i];
        }
        
        // Normalize
        for (int i = 0; i < size; i++) {
            out_row[i] /= sum;
        }
    }
}

// Matrix multiplication: C = A * B
__global__ void matmul_kernel(float* A, float* B, float* C, 
                              int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Add bias
__global__ void add_bias_kernel(float* matrix, float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / cols;
    int col = idx % cols;
    
    if (row < rows && col < cols) {
        matrix[idx] += bias[col];
    }
}

// Apply ReLU activation
__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = relu(input[idx]);
    }
}

// Forward pass for hidden layer
__global__ void forward_hidden_kernel(float* input, float* weights, float* bias,
                                     float* z, float* activation, 
                                     int batch_size, int input_dim, int output_dim) {
    int batch_idx = blockIdx.y;
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && neuron_idx < output_dim) {
        float sum = bias[neuron_idx];
        for (int i = 0; i < input_dim; i++) {
            sum += input[batch_idx * input_dim + i] * weights[i * output_dim + neuron_idx];
        }
        z[batch_idx * output_dim + neuron_idx] = sum;
        activation[batch_idx * output_dim + neuron_idx] = relu(sum);
    }
}

// Backward pass - compute output layer gradients
__global__ void backward_output_kernel(float* y_pred, float* y_true, float* grad,
                                       int batch_size, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / output_dim;
    int class_idx = idx % output_dim;
    
    if (batch_idx < batch_size && class_idx < output_dim) {
        grad[idx] = y_pred[idx] - y_true[idx];
    }
}

// Backward pass - hidden layer gradients
__global__ void backward_hidden_kernel(float* grad_next, float* weights_next,
                                       float* z, float* grad,
                                       int batch_size, int current_dim, int next_dim) {
    int batch_idx = blockIdx.y;
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && neuron_idx < current_dim) {
        float sum = 0.0f;
        for (int i = 0; i < next_dim; i++) {
            sum += grad_next[batch_idx * next_dim + i] * 
                   weights_next[neuron_idx * next_dim + i];
        }
        int idx = batch_idx * current_dim + neuron_idx;
        grad[idx] = sum * relu_derivative(z[idx]);
    }
}

// Update weights using SGD
__global__ void update_weights_kernel(float* weights, float* grad_weights,
                                     float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * grad_weights[idx];
    }
}

// Compute weight gradients
__global__ void compute_weight_gradients_kernel(float* input, float* grad_output,
                                               float* grad_weights,
                                               int batch_size, int input_dim, int output_dim) {
    int in_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (in_idx < input_dim && out_idx < output_dim) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += input[b * input_dim + in_idx] * grad_output[b * output_dim + out_idx];
        }
        grad_weights[in_idx * output_dim + out_idx] = sum / batch_size;
    }
}

// Compute bias gradients
__global__ void compute_bias_gradients_kernel(float* grad_output, float* grad_bias,
                                             int batch_size, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_dim) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += grad_output[b * output_dim + idx];
        }
        grad_bias[idx] = sum / batch_size;
    }
}

// Initialize weights using Xavier initialization
__global__ void init_weights_kernel(float* weights, unsigned long seed, 
                                   int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = (curand_uniform(&state) * 2.0f - 1.0f) * scale;
    }
}

// Cross-entropy loss
__global__ void compute_loss_kernel(float* y_pred, float* y_true, float* loss,
                                   int batch_size, int output_dim) {
    __shared__ float partial_loss[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float local_loss = 0.0f;
    if (idx < batch_size * output_dim) {
        int batch_idx = idx / output_dim;
        int class_idx = idx % output_dim;
        if (y_true[idx] > 0.5f) {
            local_loss = -logf(fmaxf(y_pred[idx], 1e-7f));
        }
    }
    
    partial_loss[tid] = local_loss;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_loss[tid] += partial_loss[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(loss, partial_loss[0]);
    }
}

// Compute accuracy
__global__ void compute_accuracy_kernel(float* y_pred, float* y_true, int* correct,
                                       int batch_size, int output_dim) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        float* pred_row = y_pred + batch_idx * output_dim;
        float* true_row = y_true + batch_idx * output_dim;
        
        int pred_class = 0;
        int true_class = 0;
        float max_pred = pred_row[0];
        
        for (int i = 1; i < output_dim; i++) {
            if (pred_row[i] > max_pred) {
                max_pred = pred_row[i];
                pred_class = i;
            }
            if (true_row[i] > 0.5f) {
                true_class = i;
            }
        }
        
        if (pred_class == true_class) {
            atomicAdd(correct, 1);
        }
    }
}

// Neural Network structure
struct NeuralNetwork {
    // Layer 1: Input to Hidden
    float *W1, *b1;
    float *dW1, *db1;
    
    // Layer 2: Hidden to Output
    float *W2, *b2;
    float *dW2, *db2;
    
    // Activations and intermediate values
    float *z1, *a1;  // Hidden layer
    float *z2, *a2;  // Output layer
    
    // Gradients
    float *grad_z2, *grad_z1;
};

void init_network(NeuralNetwork* nn) {
    // Allocate memory for weights and biases
    CUDA_CHECK(cudaMalloc(&nn->W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->b1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->b2, OUTPUT_SIZE * sizeof(float)));
    
    // Allocate memory for gradients
    CUDA_CHECK(cudaMalloc(&nn->dW1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->db1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->dW2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->db2, OUTPUT_SIZE * sizeof(float)));
    
    // Allocate memory for activations
    CUDA_CHECK(cudaMalloc(&nn->z1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->a1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->z2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->a2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Allocate memory for gradient backprop
    CUDA_CHECK(cudaMalloc(&nn->grad_z2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_z1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    
    // Initialize weights
    int blocks = (INPUT_SIZE * HIDDEN_SIZE + 255) / 256;
    float scale1 = sqrtf(6.0f / (INPUT_SIZE + HIDDEN_SIZE));
    init_weights_kernel<<<blocks, 256>>>(nn->W1, time(NULL), 
                                         INPUT_SIZE * HIDDEN_SIZE, scale1);
    
    blocks = (HIDDEN_SIZE * OUTPUT_SIZE + 255) / 256;
    float scale2 = sqrtf(6.0f / (HIDDEN_SIZE + OUTPUT_SIZE));
    init_weights_kernel<<<blocks, 256>>>(nn->W2, time(NULL) + 1, 
                                         HIDDEN_SIZE * OUTPUT_SIZE, scale2);
    
    // Initialize biases to zero
    CUDA_CHECK(cudaMemset(nn->b1, 0, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->b2, 0, OUTPUT_SIZE * sizeof(float)));
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void forward_pass(NeuralNetwork* nn, float* X, int batch_size) {
    // Layer 1: Input -> Hidden
    dim3 block1(16, 1);
    dim3 grid1((HIDDEN_SIZE + 15) / 16, batch_size);
    forward_hidden_kernel<<<grid1, block1>>>(X, nn->W1, nn->b1, nn->z1, nn->a1,
                                             batch_size, INPUT_SIZE, HIDDEN_SIZE);
    
    // Layer 2: Hidden -> Output
    dim3 block2(16, 1);
    dim3 grid2((OUTPUT_SIZE + 15) / 16, batch_size);
    forward_hidden_kernel<<<grid2, block2>>>(nn->a1, nn->W2, nn->b2, nn->z2, nn->a2,
                                             batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Apply softmax to output
    int blocks = (batch_size + 255) / 256;
    softmax_kernel<<<blocks, 256>>>(nn->z2, nn->a2, batch_size, OUTPUT_SIZE);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void backward_pass(NeuralNetwork* nn, float* X, float* y, int batch_size) {
    // Output layer gradient
    int blocks = (batch_size * OUTPUT_SIZE + 255) / 256;
    backward_output_kernel<<<blocks, 256>>>(nn->a2, y, nn->grad_z2,
                                            batch_size, OUTPUT_SIZE);
    
    // Hidden layer gradient
    dim3 block1(16, 1);
    dim3 grid1((HIDDEN_SIZE + 15) / 16, batch_size);
    backward_hidden_kernel<<<grid1, block1>>>(nn->grad_z2, nn->W2, nn->z1, nn->grad_z1,
                                              batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Compute weight gradients for W2
    dim3 block2(16, 16);
    dim3 grid2((OUTPUT_SIZE + 15) / 16, (HIDDEN_SIZE + 15) / 16);
    compute_weight_gradients_kernel<<<grid2, block2>>>(nn->a1, nn->grad_z2, nn->dW2,
                                                       batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Compute bias gradients for b2
    blocks = (OUTPUT_SIZE + 255) / 256;
    compute_bias_gradients_kernel<<<blocks, 256>>>(nn->grad_z2, nn->db2,
                                                   batch_size, OUTPUT_SIZE);
    
    // Compute weight gradients for W1
    dim3 block3(16, 16);
    dim3 grid3((HIDDEN_SIZE + 15) / 16, (INPUT_SIZE + 15) / 16);
    compute_weight_gradients_kernel<<<grid3, block3>>>(X, nn->grad_z1, nn->dW1,
                                                       batch_size, INPUT_SIZE, HIDDEN_SIZE);
    
    // Compute bias gradients for b1
    blocks = (HIDDEN_SIZE + 255) / 256;
    compute_bias_gradients_kernel<<<blocks, 256>>>(nn->grad_z1, nn->db1,
                                                   batch_size, HIDDEN_SIZE);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void update_weights(NeuralNetwork* nn, float lr) {
    int blocks;
    
    // Update W1
    blocks = (INPUT_SIZE * HIDDEN_SIZE + 255) / 256;
    update_weights_kernel<<<blocks, 256>>>(nn->W1, nn->dW1, lr, 
                                           INPUT_SIZE * HIDDEN_SIZE);
    
    // Update b1
    blocks = (HIDDEN_SIZE + 255) / 256;
    update_weights_kernel<<<blocks, 256>>>(nn->b1, nn->db1, lr, HIDDEN_SIZE);
    
    // Update W2
    blocks = (HIDDEN_SIZE * OUTPUT_SIZE + 255) / 256;
    update_weights_kernel<<<blocks, 256>>>(nn->W2, nn->dW2, lr, 
                                           HIDDEN_SIZE * OUTPUT_SIZE);
    
    // Update b2
    blocks = (OUTPUT_SIZE + 255) / 256;
    update_weights_kernel<<<blocks, 256>>>(nn->b2, nn->db2, lr, OUTPUT_SIZE);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void train(NeuralNetwork* nn, float* X_train, float* y_train, 
           int num_samples, int epochs, float lr) {
    int num_batches = (num_samples + BATCH_SIZE - 1) / BATCH_SIZE;
    
    float *d_X_batch, *d_y_batch;
    CUDA_CHECK(cudaMalloc(&d_X_batch, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_batch, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int total_correct = 0;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start = batch * BATCH_SIZE;
            int current_batch_size = (start + BATCH_SIZE <= num_samples) ? 
                                     BATCH_SIZE : (num_samples - start);
            
            // Copy batch to device
            CUDA_CHECK(cudaMemcpy(d_X_batch, X_train + start * INPUT_SIZE,
                                 current_batch_size * INPUT_SIZE * sizeof(float),
                                 cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_y_batch, y_train + start * OUTPUT_SIZE,
                                 current_batch_size * OUTPUT_SIZE * sizeof(float),
                                 cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass(nn, d_X_batch, current_batch_size);
            
            // Compute loss
            float *d_loss;
            CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
            CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
            int blocks = (current_batch_size * OUTPUT_SIZE + 255) / 256;
            compute_loss_kernel<<<blocks, 256>>>(nn->a2, d_y_batch, d_loss,
                                                 current_batch_size, OUTPUT_SIZE);
            float batch_loss;
            CUDA_CHECK(cudaMemcpy(&batch_loss, d_loss, sizeof(float),
                                 cudaMemcpyDeviceToHost));
            total_loss += batch_loss;
            
            // Compute accuracy
            int *d_correct;
            CUDA_CHECK(cudaMalloc(&d_correct, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));
            blocks = (current_batch_size + 255) / 256;
            compute_accuracy_kernel<<<blocks, 256>>>(nn->a2, d_y_batch, d_correct,
                                                     current_batch_size, OUTPUT_SIZE);
            int batch_correct;
            CUDA_CHECK(cudaMemcpy(&batch_correct, d_correct, sizeof(int),
                                 cudaMemcpyDeviceToHost));
            total_correct += batch_correct;
            
            // Backward pass
            backward_pass(nn, d_X_batch, d_y_batch, current_batch_size);
            
            // Update weights
            update_weights(nn, lr);
            
            CUDA_CHECK(cudaFree(d_loss));
            CUDA_CHECK(cudaFree(d_correct));
        }
        
        printf("Epoch %d/%d - Loss: %.4f, Accuracy: %.2f%%\n",
               epoch + 1, epochs, total_loss / num_batches,
               100.0f * total_correct / num_samples);
    }
    
    CUDA_CHECK(cudaFree(d_X_batch));
    CUDA_CHECK(cudaFree(d_y_batch));
}

void free_network(NeuralNetwork* nn) {
    CUDA_CHECK(cudaFree(nn->W1));
    CUDA_CHECK(cudaFree(nn->b1));
    CUDA_CHECK(cudaFree(nn->W2));
    CUDA_CHECK(cudaFree(nn->b2));
    CUDA_CHECK(cudaFree(nn->dW1));
    CUDA_CHECK(cudaFree(nn->db1));
    CUDA_CHECK(cudaFree(nn->dW2));
    CUDA_CHECK(cudaFree(nn->db2));
    CUDA_CHECK(cudaFree(nn->z1));
    CUDA_CHECK(cudaFree(nn->a1));
    CUDA_CHECK(cudaFree(nn->z2));
    CUDA_CHECK(cudaFree(nn->a2));
    CUDA_CHECK(cudaFree(nn->grad_z2));
    CUDA_CHECK(cudaFree(nn->grad_z1));
}