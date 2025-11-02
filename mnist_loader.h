#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Function to reverse bytes
uint32_t reverse_int(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

// Load MNIST images
float* load_mnist_images(const char* filename, int* num_images, int* num_rows, int* num_cols) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    uint32_t magic_number = 0;
    uint32_t number_of_images = 0;
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;
    
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverse_int(magic_number);
    
    fread(&number_of_images, sizeof(number_of_images), 1, file);
    number_of_images = reverse_int(number_of_images);
    
    fread(&n_rows, sizeof(n_rows), 1, file);
    n_rows = reverse_int(n_rows);
    
    fread(&n_cols, sizeof(n_cols), 1, file);
    n_cols = reverse_int(n_cols);
    
    *num_images = number_of_images;
    *num_rows = n_rows;
    *num_cols = n_cols;
    
    int image_size = n_rows * n_cols;
    float* images = (float*)malloc(number_of_images * image_size * sizeof(float));
    
    unsigned char* temp = (unsigned char*)malloc(image_size);
    
    for (int i = 0; i < number_of_images; i++) {
        fread(temp, sizeof(unsigned char), image_size, file);
        for (int j = 0; j < image_size; j++) {
            // Normalize pixel values to [0, 1]
            images[i * image_size + j] = temp[j] / 255.0f;
        }
    }
    
    free(temp);
    fclose(file);
    
    printf("Loaded %d images of size %dx%d\n", number_of_images, n_rows, n_cols);
    return images;
}

// Load MNIST labels
float* load_mnist_labels(const char* filename, int* num_labels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    uint32_t magic_number = 0;
    uint32_t number_of_labels = 0;
    
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverse_int(magic_number);
    
    fread(&number_of_labels, sizeof(number_of_labels), 1, file);
    number_of_labels = reverse_int(number_of_labels);
    
    *num_labels = number_of_labels;
    
    // One-hot encode labels
    float* labels = (float*)calloc(number_of_labels * 10, sizeof(float));
    unsigned char* temp = (unsigned char*)malloc(number_of_labels);
    
    fread(temp, sizeof(unsigned char), number_of_labels, file);
    
    for (int i = 0; i < number_of_labels; i++) {
        labels[i * 10 + temp[i]] = 1.0f;
    }
    
    free(temp);
    fclose(file);
    
    printf("Loaded %d labels\n", number_of_labels);
    return labels;
}

#endif // MNIST_LOADER_H