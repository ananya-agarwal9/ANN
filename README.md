#STL-10
The STL-10 dataset, in contrast, contains colored images of size 96x96 pixels and poses a greater challenge due to its higher resolution and complexity. It is specifically designed for tasks involving unsupervised feature learning and image classification, with a fixed split of 5,000 training images and 8,000 test images distributed equally across the 10 classes.
1. Input Layer
Input Shape: (96, 96, 3) for RGB images
2. Convolutional Layers
Convolutional Layer 1:
Number of Filters: 64
Filter Size: 3x3
Stride: 1
Padding: 1 (Same Padding)
Activation Function: ReLU
Batch Normalization: Applied after this layer
Convolutional Layer 2:
Number of Filters: 128
Filter Size: 3x3
Stride: 1
Padding: 1 (Same Padding)
Activation Function: ReLU
Batch Normalization: Applied after this layer
Convolutional Layer 3:
Number of Filters: 256
Filter Size: 3x3
Stride: 1
Padding: 1 (Same Padding)
Activation Function: ReLU
Batch Normalization: Applied after this layer
Convolutional Layer 4:
Number of Filters: 512
Filter Size: 3x3
Stride: 1
Padding: 1 (Same Padding)
Activation Function: ReLU
Batch Normalization: Applied after this layer
3. Pooling Layer
Type: Max Pooling
Pool Size: 2x2
Stride: 2
4. Global Average Pooling
Converts feature maps into a vector by computing the average of each feature map.
5. Dropout Layer
Dropout Rate: 0.5 (50% of neurons are dropped)
6. Fully Connected Layers
Fully Connected Layer 1:
Units: 1024
Activation Function: ReLU
Fully Connected Layer 2:
Units: 512
Activation Function: ReLU
Fully Connected Layer 3:
Units: 10 (One for each class in STL-10)
Activation Function: Softmax
7. Optimization Details
Optimizer: Adam
Loss Function: Categorical Crossentropy
Evaluation Metrics: Accuracy
Learning Rate: 0.001
Mini-Batch Size: 64
Epochs: 50
Total Iterations: n×epochs=64×50
