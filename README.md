# Airplane-Classification

Classification for airplane families written in PyTorch.

## Models Tried
Permutations of the following
- 1,2,3 Convolutional layers with kernels of size 3,5
- 1,2 Fully Connected Layers
- Dropouts, BatchNorm
- MaxPool, AveragePool
- SGD, Adam optimizer

## Results

### Model
- 2 Convolutional layers with 2 Fully connected layers, SGD optimizer
- Params: 2.5M
- Overall val accuracy: ~40%
- Classwise accuracy in the jupyter notebook

### Observations
- Low accuracy can be explained from low image resolution (107x72), less number of training samples (3334) wrt to number of classes (70).
- Deep networks don't perform better than shallow ones because of less resolution
- SOTA models like ResNet, Inception, VGG need atleast image resolution to be (224,224), hence unable to test any pretrained models also.
- Airplane families aren't very distinguishable from each other for human classification from these images

### Future directions for improvement
- Models with residual learning, involving identity function from previous layers can be explored.
- Adding more training data.
