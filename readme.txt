## Perceptual Lossy Compression

This the code for the paper "On Perceptual Lossy Compression: The Cost of Perceptual Reconstruction and An Optimal Training Framework, ICML, 2021"

## Release notes

This repository is a faithful reimplementation of (https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN) in PyTorch

* you can download the dataset from
  - MNIST dataset: http://yann.lecun.com/exdb/mnist/

## Development Environment

* Win10
* NVIDIA GeForce RTX 2080
* cuda version 11.1
* Python 3.6
* pytorch 1.2.0
* torchvision 0.4.0
* matplotlib 3.1.2

# Mode definition:
networks.py

# Training code:
python train_MNIST.py

# Testing code
python test_MNIST.py

# Parameters can be modified in train_MNIST.py or test_MNIST.py in the section：
-- training parameters
config = get_parameters()
batch_size = 128
lr = 0.001
train_epoch = 2
lambda_gp = 10
pretrained = False
rate = 4          # bit-rate setting
img_size = 32

--testing parameters
batch_size = 100
rate = 4
img_size = 32

## Citation
@inproceedings{2021PerceptualReconstruction,
	author={Zeyu Yan, Fei Wen, Rendong Ying, Chao Ma, Peilin Liu},
	booktitle={Proceedings of the
	International Conference on Machine Learning (ICML)},
	title={On Perceptual Lossy Compression: The Cost of Perceptual Reconstruction and An Optimal Training Framework},
	year={2021}, 
}