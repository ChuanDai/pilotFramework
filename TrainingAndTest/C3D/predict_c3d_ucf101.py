from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf
import input_data
import c3d_model
import numpy as np

flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
FLAGS = flags.FLAGS


def placeholder_inputs(batch_size):
    """
    Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop.

    :param batch_size: The batch size will be baked into both placeholders.

    :return images_placeholder: Images placeholder.
    :return labels_placeholder: Labels placeholder.
    """
    # note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    return images_placeholder, labels_placeholder


def _variable_on_cpu(name, shape, initializer):
    """
    Retrieves a variable with the given name and shape on the CPU
    by the specified initialization method.

    :param name: Name.
    :param shape: Shape.
    :param initializer: Initialization method.
    :return var: The variable.
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Applying truncated normal initialization method on the CPU to get the variable
    with the given name and shape, and its L2 regularization term is saved.

    :param name: Name.
    :param shape: Shape.
    :param wd: L2 regularization coefficient.

    :return var: The variable.
    """
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('losses', weight_decay)
    return var


def run_test():
    # gets the path to the model file to be tested.
    model_name = "./C3D-tensorflow/sports1m_finetuning_ucf101.model"
    # get the list file that holds the test set information
    test_list_file = 'test.list'
    # printing the amount of test data.
    num_test_videos = len(list(open(test_list_file, 'r')))
    print("Number of test videos={}".format(num_test_videos))

    # initializing a batch sample of the input model corresponding to the tensor
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
    with tf.variable_scope('var_name') as var_scope:
        # based on the definition of C3D network structure
        # Learning Spatiotemporal Features with 3D Convolutional Networks
        # https://arxiv.org/pdf/1412.0767.pdf
        weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
            }
        biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0)
            }
    logits = []
    # traverse the GPU, distribute a batch data equally to each GPU for testing,
    # and add the results of network output to the list.
    for gpu_index in range(0, gpu_num):
        with tf.device('/gpu:%d' % gpu_index):
            logit = c3d_model.inference_c3d(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size, :, :, :, :], 0.6, FLAGS.batch_size, weights, biases)
            logits.append(logit)
    # calculating the softmax score of the network output.
    logits = tf.concat(logits, 0)
    norm_score = tf.nn.softmax(logits)
    # creating a Saver object that holds all the variables.
    saver = tf.train.Saver()
    # creating a session that runs all the operations in the calculation diagram
    # and automatically selecting the device to run on.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    # initializing all variables.
    sess.run(init)
    # loading the model to test.
    saver.restore(sess, model_name)
    # open predict_ret.txt.
    # https://stackoverflow.com/questions/45263064/how-can-i-fix-this-valueerror-cant-have-unbuffered-text-i-o-in-python-3/45263101
    # bufsize = 0
    # write_file = open("predict_ret.txt", "w+", bufsize)
    write_file = open("predict_ret.txt", "w+")
    next_start_pos = 0
    # counting test times
    all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)
    # get a batch of test samples in order for testing each time,
    # and get the probability and prediction category
    # of each batch test sample in each category.
    # writing the information for each test sample to predict_ret.txt.

    # the format of the information written to predict_ret.txt is
    # [ground truth][class probability for true label][predicted label][class probability for predicted label]
    for step in xrange(all_steps):
        # fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        test_images, test_labels, next_start_pos, _, valid_len = \
            input_data.read_clip_and_label(
                    test_list_file,
                    FLAGS.batch_size * gpu_num,
                    start_pos=next_start_pos
                    )
        predict_score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: test_images}
            )
        for i in range(0, valid_len):
            true_label = test_labels[i],
            top1_predicted_label = np.argmax(predict_score[i])
            # write results to predict_ret.txt
            write_file.write('{}, {}, {}, {}\n'.format(
                # ground truth
                true_label[0],
                # class probability for true label
                predict_score[i][true_label],
                # predicted label
                top1_predicted_label,
                # class probability for predicted label
                predict_score[i][top1_predicted_label])
            )
    write_file.close()
    print("done")


def main(_):
    run_test()


if __name__ == '__main__':
    tf.app.run()
