import tensorflow as tf
from PIL import Image, ImageFilter


def predictint(imvalue):
    """

    This function returns the predicted integer.

    The imput is the pixel values from the imageprepare() function.

    """

    # Define the model (same as when creating the model file)

    # Convolutional Layer 1
    filter_size1 = 5
    num_filters1 = 16

    # Convolutional Layer 2
    filter_size2 = 5
    num_filters2 = 36

    # Fully-connected layer
    fc_size = 128

    # We know that MNIST images are 28 pixels in each dimension.
    img_size = 28

    # Images are stored in one-dimensional array of this length.
    img_size_flat = img_size * img_size

    # Tuple with height and width of images used to reshape arrays.

    # Number of colour channels for the images: 1 channel for gray-scale.
    num_channels = 1

    # Number of classes, one class for each of 10 digits.
    num_classes = 10

    def new_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def new_conv_layer(input,  # The previous layer.
                       num_input_channels,  # Num. channels in prev. layer.
                       filter_size,  # Width and height of each filter.
                       num_filters,  # Number of filters.
                       use_pooling=True):  # Use 2x2 max-pooling.

        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights aka. filters with the given shape.
        weights = new_weights(shape=shape)

        # Create new biases, one for each filter.
        biases = new_biases(length=num_filters)

        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases

        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)

        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights

    def flatten_layer(layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]

        # Return both the flattened layer and the number of features.
        return layer_flat, num_features

    def new_fc_layer(input,  # The previous layer.
                     num_inputs,  # Num. inputs from prev. layer.
                     num_outputs,  # Num. outputs.
                     use_relu=True):  # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = new_weights(shape=[num_inputs, num_outputs])
        biases = new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

    layer_conv1, weights_conv1 = \
        new_conv_layer(input=x_image,
                       num_input_channels=num_channels,
                       filter_size=filter_size1,
                       num_filters=num_filters1,
                       use_pooling=True)

    layer_conv2, weights_conv2 = \
        new_conv_layer(input=layer_conv1,
                       num_input_channels=num_filters1,
                       filter_size=filter_size2,
                       num_filters=num_filters2,
                       use_pooling=True)

    layer_flat, num_features = flatten_layer(layer_conv2)

    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             use_relu=True)

    layer_fc2 = new_fc_layer(input=layer_fc1,
                             num_inputs=fc_size,
                             num_outputs=num_classes,
                             use_relu=False)

    y_pred = tf.nn.softmax(layer_fc2)

    y_pred_cls = tf.argmax(y_pred, axis=1)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    """

    Load the model2.ckpt file

    file is stored in the same directory as this python script is started

    Use the model to predict the integer. Integer is returned as list.


    Based on the documentation at

    https://www.tensorflow.org/versions/master/how_tos/variables/index.html

    """

    with tf.Session() as sess:
        sess.run(init_op)

        saver.restore(sess, "model2.ckpt")

        print("Model restored.")

        feed_dict1 = {x: [imvalue]}
        classification = sess.run(y_pred_cls, feed_dict1)
        return classification


def imageprepare(argv):
    """

    This function returns the pixel values.

    The input is a png file location.

    """

    im = Image.open(argv).convert('L')

    width = float(im.size[0])

    height = float(im.size[1])

    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger

        # Width is bigger. Width becomes 20 pixels.

        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width

        if (nheigth == 0):  # rare case but minimum is 1 pixel

            nheigth = 1

            # resize and sharpen

        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

        wtop = int(round(((28 - nheight) / 2), 0))  # caculate horizontal pozition

        newImage.paste(img, (4, wtop))  # paste resized image on white canvas

    else:

        # Height is bigger. Heigth becomes 20 pixels.

        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height

        if (nwidth == 0):  # rare case but minimum is 1 pixel

            nwidth = 1

            # resize and sharpen

        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical position

        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

        newImage.save("sample.png")

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.

    tva = [(255 - x) * 1.0 / 255.0 for x in tv]

    return tva


def main(argv):
    """

    Main function.

    """

    imvalue = imageprepare(argv)

    predint = predictint(imvalue)

    print(predint[0])  # first value in list



main('img/1.png')
