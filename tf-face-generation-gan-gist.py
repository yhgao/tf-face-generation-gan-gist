# This is a gist of a GAN (generative adversarial networks) I built during the deep learning foundations course

# First, using MNIST (will show how well the model trains sooner)
#   and then CelebA datasets. CelebA has 200,000+ celebrity images with annotations.
# The values of the MNIST and CelebA datasets used in this project were in the range of -0.5 to 0.5 of 28x28 dimensional images; the CelebA images were cropped to remove parts of the image that don't include a face; the images were also resized to 28x28.


# You can find the generated celebrity faces are in: celeba-realistic-faces-test.png

# Using TensorFlow and cloud GPUs

import tensorflow as tf


# *** Build the components necessary to build the GANs ***
# Functions to be implemented
# - model_inputs
# - discriminator
# - generator
# - model_loss
# - model_opt
# - train

# Input
# Implement the model_inputs function to create TF placeholders
# where we have:
# - Real input images placeholder with rank 4 using image_width, image_height, and image_channels.
# - Z input placeholder with rank 2 using z_dim.
# - Learning rate placeholder with rank 0.
# - Return the placeholders in the following the tuple (tensor of real input images, tensor of z data)

def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs
    :param image_width: The input image width
    :param image_height: The input image height
    :param image_channels: The number of image channels
    :param z_dim: The dimension of Z
    :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
    """

    # Real images
    real_inputs = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name = 'real_inputs')

    # Uniform noise distribution Z (vector going into the generator)
    z_inputs = tf.placeholder(tf.float32, (None, z_dim), name = 'z_inputs')

    # Learning rate
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return real_inputs, z_inputs, learning_rate


# Discriminator
# Implement discriminator to create a discriminator neural net that discriminates on images.
# Use tf.variable_scope with a scope name of "discriminator" to allow the variables to be reused.
# Return a tuple of (tensor output of the discriminator, tensor logits of the discriminator).

def discriminator(images, reuse=False):
    """
    Create the discriminator network
    :param image: Tensor of input image(s)
    :param reuse: Boolean if the weights should be reused
    :return: Tuple of (tensor output of the discriminator, tensor logits of the discriminator)
    """
    with tf.variable_scope('discriminator', reuse = reuse):

        alpha = 0.2

        # Input layer 28x28x3 (image is 28x28), note there is no batch norm on the 1st layer
        x1 = tf.layers.conv2d(inputs = images, filters = 64, kernel_size = 5, strides = 2, padding = 'same')
        # Leaky ReLU
        x1 = tf.maximum(alpha * x1, x1)
        # 14x14x64 conv layer

        # Next layer
        x2 = tf.layers.conv2d(inputs = x1, filters = 128, kernel_size = 5, strides = 2, padding = 'same')
        # Batch normalization "stabilizes learning by normalizing the input to each unit to have zero mean and unit variance. This helps deal with training problems that arise due to poor initialization and helps gradient flow in deeper models." (https://arxiv.org/pdf/1511.06434.pdf)"
        x2 = tf.layers.batch_normalization(x2, training = True)
        # LReLU activation
        x2 = tf.maximum(alpha * x2, x2)
        # 7x7x128 con layer

        # Next layer
        x3 = tf.layers.conv2d(inputs = x2, filters = 256, kernel_size = 5, strides = 2, padding = 'same')
        # Batch norm
        x3 = tf.layers.batch_normalization(x3, training = True)
        # LReLU activation
        x3 = tf.maximum(alpha * x3, x3)
        # 4x4x256 con layer

        # Flattening
        flat = tf.reshape(x3, (-1, 4 * 4 * 256))
        # Finally, logits
        logits = tf.layers.dense(flat, 1)
        # and output
        out = tf.sigmoid(logits)

    # Tensor output and logits of the discriminator
    return out, logits


# Generator
# Implement generator to generate an image using z.
# Use tf.variable_scope with a scope name of "generator" to allow the variables to be reused.
# Return the generated 28 x 28 x out_channel_dim images.

def generator(z, out_channel_dim, is_train=True):
    """
    Create the generator network
    :param z: Input z
    :param out_channel_dim: The number of channels in the output image
    :param is_train: Boolean if generator is being used for training
    :return: The tensor output of the generator
    """

    with tf.variable_scope('generator', reuse = not is_train):

        alpha = 0.2

        # Fully connected layer, depth 512
        x1 = tf.layers.dense(z, 2 * 2 * 512)
        # Reshape it to the start the conv stack
        x1 = tf.reshape(x1, (-1, 2, 2, 512))
        # Batch norm
        x1 = tf.layers.batch_normalization(x1, training = is_train)
        # LReLU
        x1 = tf.maximum(alpha * x1, x1)

        # Next layer, depth 256. Notice padding is 'valid'
        x2 = tf.layers.conv2d_transpose(inputs = x1, filters = 256, kernel_size = 5, strides = 2, padding = 'valid')
        # Batch norm
        x2 = tf.layers.batch_normalization(x2, training = is_train)
        # LReLU
        x2 = tf.maximum(alpha * x2, x2)

        # Next layer, depth 128
        x3 = tf.layers.conv2d_transpose(inputs = x2, filters = 128, kernel_size = 5, strides = 2, padding = 'same')
        # Batch norm
        x3 = tf.layers.batch_normalization(x3, training = is_train)
        # LReLU
        x3 = tf.maximum(alpha * x3, x3)

        # Use the last layer as logits
        logits = tf.layers.conv2d_transpose(inputs = x3, filters = out_channel_dim, kernel_size = 5, strides = 2, padding = 'same')
        # 28x28x3 - and the image is 28x28
        # and tanh output
        out = tf.tanh(logits)

    # Tensor output of the generator
    return out


# Loss
# Implement model_loss to build the GANs for training and calculate the loss.
# Return a tuple of (discriminator loss, generator loss).
# Use
# - discriminator(images, reuse=False)
# - generator(z, out_channel_dim, is_train=True)

def model_loss(input_real, input_z, out_channel_dim):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    g_model = generator(input_z, out_channel_dim)
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse = True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real,
        labels=tf.ones_like(d_model_real)))

    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits = d_logits_fake,
        labels = tf.zeros_like(d_model_fake)))

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits = d_logits_fake,
        labels = tf.ones_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake

    # Discriminator and generator losses
    return d_loss, g_loss


# Optimisation
# Implement model_opt to create the optimisation operations for the GANs.
# Use tf.trainable_variables to get all the trainable variables.
# Filter the variables with names that are in the discriminator and generator scope names.
# Return a tuple of (discriminator training operation, generator training operation).

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimiser
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimise
    d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(d_loss, var_list = d_vars)
    ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    g_updates = [opt for opt in ops if opt.name.startswith('generator')]

    with tf.control_dependencies(g_updates):
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1).minimize(g_loss, var_list = g_vars)

    # Discriminator/generator training operations
    return d_train_opt, g_train_opt


# *** Training ***
# Show generator output
import numpy as np

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()


# Train GANs
# Use model_inputs(image_width, image_height, image_channels, z_dim), model_loss(input_real, input_z, out_channel_dim), model_opt(d_loss, g_loss, learning_rate, beta1)
# Running show_generator_output for every batch will drastically increase training time

def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimiser
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    # Data shape
    _, img_width, img_height, img_channels = data_shape

    # Inputs
    real_input, z_input, lr = model_inputs(img_width, img_height, img_channels, z_dim)

    # Loss (discriminator, generator)
    d_loss, g_loss = model_loss(real_input, z_input, img_channels)

    # Optimisers (discriminator, generator)
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)

    print_every = 10
    show_every = 100
    n_images = 25
    losses = []

    steps = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):

                batch_images *= 2.0

                steps += 1

                # 1) Sample random noise for G
                z_sample = np.random.uniform(-1, 1, (batch_size, z_dim))

                # 2) Run optimisers
                _ = sess.run(d_opt, feed_dict = {real_input: batch_images, z_input: z_sample, lr: learning_rate})
                _ = sess.run(g_opt, feed_dict = {z_input: z_sample, lr: learning_rate})

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = d_loss.eval({z_input: z_sample, real_input: batch_images})
                    train_loss_g = g_loss.eval({z_input: z_sample})

                    print("Epoch {}/{}...".format(epoch_i+1, epoch_count), "Discriminator Loss: {:.4f}...".format(train_loss_d), "Generator Loss: {:.4f}".format(train_loss_g))
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))

                if steps % show_every == 0:
                    show_generator_output(sess, n_images, z_input, img_channels, data_image_mode)

# Test on MNIST
batch_size = 32
z_dim = 100
learning_rate = 0.01
beta1 = 0.5
epochs = 2

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)

# Epoch 1/2... Discriminator Loss: 0.0023... Generator Loss: 15.6878
# Epoch 1/2... Discriminator Loss: 0.0001... Generator Loss: 12.6037

# ...

# Epoch 2/2... Discriminator Loss: 1.3243... Generator Loss: 0.5069
# Epoch 2/2... Discriminator Loss: 0.6219... Generator Loss: 1.0231
# Epoch 2/2... Discriminator Loss: 2.1300... Generator Loss: 0.2804


# Run GANs on CelebA
batch_size = 32
z_dim = 100
learning_rate = 0.005
beta1 = 0.5
epochs = 1

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)

# Epoch 1/1... Discriminator Loss: 1.1708... Generator Loss: 1.8845
# Epoch 1/1... Discriminator Loss: 2.6645... Generator Loss: 0.4742
# Epoch 1/1... Discriminator Loss: 0.0905... Generator Loss: 4.0524
# Epoch 1/1... Discriminator Loss: 3.3082... Generator Loss: 10.7925
# Epoch 1/1... Discriminator Loss: 0.7239... Generator Loss: 1.0459

# ...

# Epoch 1/1... Discriminator Loss: 1.0916... Generator Loss: 0.9408
# Epoch 1/1... Discriminator Loss: 1.2753... Generator Loss: 0.6810
# Epoch 1/1... Discriminator Loss: 1.3050... Generator Loss: 0.6114
# Epoch 1/1... Discriminator Loss: 2.0163... Generator Loss: 0.1833
# Epoch 1/1... Discriminator Loss: 1.0962... Generator Loss: 1.0769
# Epoch 1/1... Discriminator Loss: 1.8095... Generator Loss: 0.2543
# Epoch 1/1... Discriminator Loss: 1.1721... Generator Loss: 0.5124
# Epoch 1/1... Discriminator Loss: 1.6530... Generator Loss: 0.2793

# ...

# The generated celebrity faces are IN celeba-realistic-faces-test.png
