#include "mnist_loader.h"
#include "neural_network.cu"

int main() {
    printf("CUDA Neural Network - MNIST Digit Classification\n");
    printf("==================================================\n\n");
    
    // Check CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA device(s)\n", device_count);
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n\n", prop.major, prop.minor);
    
    // Load MNIST data
    printf("Loading MNIST dataset...\n");
    int num_train_images, num_test_images, num_rows, num_cols;
    int num_train_labels, num_test_labels;
    
    float* X_train = load_mnist_images("train-images-idx3-ubyte", 
                                       &num_train_images, &num_rows, &num_cols);
    float* y_train = load_mnist_labels("train-labels-idx1-ubyte", &num_train_labels);
    
    float* X_test = load_mnist_images("t10k-images-idx3-ubyte",
                                      &num_test_images, &num_rows, &num_cols);
    float* y_test = load_mnist_labels("t10k-labels-idx1-ubyte", &num_test_labels);
    
    if (!X_train || !y_train || !X_test || !y_test) {
        fprintf(stderr, "Failed to load MNIST dataset\n");
        return 1;
    }
    
    printf("\nDataset loaded successfully!\n");
    printf("Training samples: %d\n", num_train_images);
    printf("Test samples: %d\n\n", num_test_images);
    
    // Initialize neural network
    printf("Initializing neural network...\n");
    printf("Architecture: %d -> %d -> %d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Batch size: %d\n", BATCH_SIZE);
    printf("Learning rate: %.4f\n", LEARNING_RATE);
    printf("Epochs: %d\n\n", EPOCHS);
    
    NeuralNetwork nn;
    init_network(&nn);
    
    // Train the network
    printf("Starting training...\n");
    printf("==================================================\n");
    train(&nn, X_train, y_train, num_train_images, EPOCHS, LEARNING_RATE);
    printf("==================================================\n");
    printf("Training completed!\n\n");
    
    // Evaluate on test set
    printf("Evaluating on test set...\n");
    int test_batches = (num_test_images + BATCH_SIZE - 1) / BATCH_SIZE;
    int total_correct = 0;
    
    float *d_X_test, *d_y_test;
    CUDA_CHECK(cudaMalloc(&d_X_test, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_test, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    for (int batch = 0; batch < test_batches; batch++) {
        int start = batch * BATCH_SIZE;
        int current_batch_size = (start + BATCH_SIZE <= num_test_images) ?
                                 BATCH_SIZE : (num_test_images - start);
        
        CUDA_CHECK(cudaMemcpy(d_X_test, X_test + start * INPUT_SIZE,
                             current_batch_size * INPUT_SIZE * sizeof(float),
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y_test, y_test + start * OUTPUT_SIZE,
                             current_batch_size * OUTPUT_SIZE * sizeof(float),
                             cudaMemcpyHostToDevice));
        
        forward_pass(&nn, d_X_test, current_batch_size);
        
        int *d_correct;
        CUDA_CHECK(cudaMalloc(&d_correct, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));
        
        int blocks = (current_batch_size + 255) / 256;
        compute_accuracy_kernel<<<blocks, 256>>>(nn.a2, d_y_test, d_correct,
                                                 current_batch_size, OUTPUT_SIZE);
        
        int batch_correct;
        CUDA_CHECK(cudaMemcpy(&batch_correct, d_correct, sizeof(int),
                             cudaMemcpyDeviceToHost));
        total_correct += batch_correct;
        
        CUDA_CHECK(cudaFree(d_correct));
    }
    
    printf("Test Accuracy: %.2f%%\n", 100.0f * total_correct / num_test_images);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_X_test));
    CUDA_CHECK(cudaFree(d_y_test));
    free_network(&nn);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);
    
    printf("\nProgram completed successfully!\n");
    return 0;
}