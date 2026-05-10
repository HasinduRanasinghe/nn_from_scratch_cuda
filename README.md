# NN From Scratch (CUDA)

A minimal two-layer neural network implemented in CUDA for MNIST digit classification.

**Overview**
- This project implements a simple feed-forward neural network (input -> hidden -> output) and trains it on the MNIST dataset entirely on the GPU using CUDA kernels.
- It is intended as a learning / demonstration project for CUDA-based neural network primitives (matrix multiply, activations, softmax, backprop, SGD).

**Repository Structure**
- [main.cu](main.cu): Program entry, dataset loading, training loop, evaluation.
- [neural_network.cu](neural_network.cu): CUDA kernels and neural network implementation (forward, backward, weight updates, training helpers).
- [mnist_loader.h](mnist_loader.h): Simple MNIST binary file loader and one-hot label encoder.

**Prerequisites**
- NVIDIA GPU with CUDA support and a working CUDA Toolkit (nvcc).
- `nvcc` on PATH (from the CUDA Toolkit).
- MNIST dataset files (binary IDX format), placed in the working directory with the following filenames:
  - `train-images-idx3-ubyte`
  - `train-labels-idx1-ubyte`
  - `t10k-images-idx3-ubyte`
  - `t10k-labels-idx1-ubyte`

Download MNIST from http://yann.lecun.com/exdb/mnist/ and place the files in the project directory before running.

**Build**
From the project root run (adjust `-arch` for your GPU if desired):

Linux / macOS:
```
nvcc -O2 -arch=sm_60 -o nn main.cu -lcurand
```

Windows (PowerShell):
```
nvcc -O2 -arch=sm_60 -o nn.exe main.cu -lcurand
```

Notes:
- `neural_network.cu` is #included by `main.cu`, so compiling `main.cu` builds the whole project.
- `-lcurand` is required because the code uses `curand` for weight initialization.

**Run**
Place the MNIST IDX files in the same directory and run:

Linux / macOS:
```
./nn
```

Windows (PowerShell):
```
.\nn.exe
```

The program prints device info, dataset loading progress, training progress per epoch (loss and accuracy), and final test accuracy.

**Default Hyperparameters**
These are set as `#define` macros in `neural_network.cu` and can be adjusted before compiling:
- `INPUT_SIZE` = 784 (28x28 flattened)
- `HIDDEN_SIZE` = 128
- `OUTPUT_SIZE` = 10
- `BATCH_SIZE` = 64
- `LEARNING_RATE` = 0.01
- `EPOCHS` = 10

**What the code does (brief)**
- Loads MNIST images and one-hot encodes labels (`mnist_loader.h`).
- Initializes a two-layer network on the GPU (Xavier-style initialization using curand).
- Implements forward pass kernels for hidden and output layers, softmax, loss and accuracy computation.
- Implements backward pass kernels to compute gradients and kernels to update weights with SGD.
- Trains for the configured number of epochs and evaluates accuracy on the test set.

**Limitations & Notes**
- This is a pedagogical implementation and is not optimized for speed or numerical stability like production frameworks.
- The code uses fixed-size batches allocated according to `BATCH_SIZE`; if your dataset size is not a multiple of the batch size the last batch is handled but temporary device buffers use `BATCH_SIZE` sized allocations.
- Adjust `-arch` in the `nvcc` command to match your GPU compute capability (e.g., `sm_75`, `sm_80`).
- If you encounter out-of-memory errors, reduce `BATCH_SIZE` or `HIDDEN_SIZE`.

**Possible Improvements**
- Add momentum or Adam optimizer.
- Use cuBLAS for matrix multiplications for better performance.
- Implement mini-batch shuffling, better memory reuse, and stream overlap for data transfer.
- Add command-line flags to control hyperparameters and file paths.
