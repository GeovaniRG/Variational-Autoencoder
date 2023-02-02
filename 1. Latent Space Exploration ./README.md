Here's a code example for sampling from the latent space and generating new images using the trained VAE: 

This code generates a random latent code with the specified number of dimensions (latent_dim), passes it through the decoder, and then displays the generated image. Note that the values of the generated image are first clipped to the [0, 1] range before display.

```
# Sample a latent code
latent_code = torch.randn(1, latent_dim)

# Pass the latent code through the decoder to generate an image
reconstructed_image = model.decoder(latent_code)

# Clip the image values to the [0, 1] range and convert to numpy for display
reconstructed_image = reconstructed_image.clamp(0, 1).detach().numpy()[0]

# Display the generated image
plt.imshow(reconstructed_image, cmap='gray')
plt.show()

```

The code for sampling from the latent space and generating new images consists of several steps:

1- Sampling a latent code: A random latent code is generated with torch.randn(1, latent_dim), where latent_dim is the number of dimensions in the latent space. This code is essentially a vector of random numbers that will be used to generate a new image.

2- Passing the latent code through the decoder: The generated latent code is then passed through the decoder model.decoder(latent_code) to generate a new image. The decoder takes the latent code as input and outputs a reconstructed image.

3- Clipping and converting the image: Before display, the values of the generated image are clipped to the range [0, 1] using clamp(0, 1). This is done because the pixel values in the MNIST dataset are in the range [0, 1], and values outside this range might not be suitable for display. The image is then converted to a numpy array using detach().numpy()[0] so that it can be displayed using the imshow function from the matplotlib library.

4- Displaying the generated image: Finally, the generated image is displayed using plt.imshow(reconstructed_image, cmap='gray') and plt.show(). The cmap='gray' argument is used to specify that the image should be displayed in grayscale.

With these steps, you can generate and display new images using the trained VAE.
