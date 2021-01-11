import tensorflow as tf

# UCF-101 dataset has 101 classes
NUM_CLASSES = 101

# images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
CHANNELS = 3

# number of frames per video clip
NUM_FRAMES_PER_CLIP = 16


def conv3d(name, l_input, w, b):
    """
    A 3D convolution operation named name is defined: the 3D convolution kernel W
    is used to convolve the 3D input L_input
    (Note: the sliding step size of the convolution kernel in each dimension is 1,
    and the output 3D feature map and the size of the 3D input are the same),
    and a bias b is added.

    :param name: The name of 3D convolution operation.
    :param l_input: 3D input
    :param w: weight
    :param b: bias
    :return: A tensor with the same type as value.
    """
    return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
          b
    )


def max_pool(name, l_input, k):
    """
    Define a maximum 3D pooling operation named name for the l_input as input
    (Note: the pooling core has a sliding step of k on dimension 1,
    a sliding step of 2 on dimensions 2 and 3,
    and a sliding step of 1 on other dimensions, filling in with 0 if necessary).

    :param name: The name of 3D pooling operation.
    :param l_input: 3D input.
    :param k: Sliding stride.
    :return: A max pooled output tensor.
    """
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)


def inference_c3d(_X, _dropout, batch_size, _weights, _biases):
    """
    Build the overall framework of C3D network.

    :param _X: The tensor of network input.
    :param _dropout: The drop rate.
    :param batch_size: Number of samples used per iteration while network training.
    :param _weights: A dictionary that holds weight parameter configuration information.
    :param _biases: A dictionary that holds bias parameter configuration information.

    :return: A tensor for network classification layer output.
    """
    # convolution layer
    conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
    conv1 = tf.nn.relu(conv1, 'relu1')
    pool1 = max_pool('pool1', conv1, k=1)

    # convolution layer
    conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
    conv2 = tf.nn.relu(conv2, 'relu2')
    pool2 = max_pool('pool2', conv2, k=2)

    # convolution layer
    conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
    conv3 = tf.nn.relu(conv3, 'relu3a')
    conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
    conv3 = tf.nn.relu(conv3, 'relu3b')
    pool3 = max_pool('pool3', conv3, k=2)

    # convolution layer
    conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
    conv4 = tf.nn.relu(conv4, 'relu4a')
    conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
    conv4 = tf.nn.relu(conv4, 'relu4b')
    pool4 = max_pool('pool4', conv4, k=2)

    # convolution layer
    conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
    conv5 = tf.nn.relu(conv5, 'relu5a')
    conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
    conv5 = tf.nn.relu(conv5, 'relu5b')
    pool5 = max_pool('pool5', conv5, k=2)

    # fully connected layer
    pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
    # reshape conv3 output to fit dense layer input
    dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

    # relu activation
    dense1 = tf.nn.relu(dense1, name='fc1')
    # https://stackoverflow.com/questions/55235230/tensorflow-please-use-rate-instead-of-keep-prob-rate-should-be-set-to-rat
    dense1 = tf.nn.dropout(dense1, rate = 1- _dropout)
    # dense1 = tf.nn.dropout(dense1, _dropout)

    # relu activation
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')
    # https://stackoverflow.com/questions/55235230/tensorflow-please-use-rate-instead-of-keep-prob-rate-should-be-set-to-rat
    dense2 = tf.nn.dropout(dense2, rate = 1- _dropout)
    # dense2 = tf.nn.dropout(dense2, _dropout)

    # class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']

    return out
