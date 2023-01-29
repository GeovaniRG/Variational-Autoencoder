# Variational-Autoencoder

his code implements a Variational Autoencoder (VAE), which is a type of generative model that can learn a compact representation of the data by encoding it into a lower-dimensional latent space and then decoding it back to its original form. The goal of the VAE is to minimize the reconstruction loss, which measures the difference between the input image and the reconstructed image, and also to encourage the encoded representation to have a Gaussian distribution. This is achieved by adding a regularization term to the loss function that penalizes deviation from a Gaussian distribution, which is the Kullback-Leibler divergence between the encoding distribution and the standard Gaussian.

The model architecture consists of an encoding part, a reparameterization step, and a decoding part. The encoding part takes the input image and maps it to a pair of mean and log variance, which are then used to sample a latent code via the reparameterization step. The latent code is then passed through the decoding part to reconstruct the image. The architecture is defined in the VAE class, which subclasses nn.Module and overrides its forward method to define the forward pass of the model.

The MNIST dataset is loaded using the torchvision library, and the data is transformed to tensors and normalized to have a mean of 0.5 and a standard deviation of 0.5. The transformed data is then loaded using the PyTorch data loader.

The model is trained using the Adam optimizer, which adjusts the model parameters in the direction that minimizes the loss. During training, the model processes the training data in batches, computes the reconstruction loss, and updates the parameters using backpropagation. The training progress is reported every epoch, showing the average training loss over the entire training set.

* Variational_Autoencoder.py

Here's an example code for training a Variational Autoencoder (VAE) on the MNIST dataset using PyTorch.
