import os
import time
from six.moves import xrange
import tensorflow as tf
import input_data
import c3d_model


flags = tf.app.flags
gpu_num = 1
# flags.DEFINE_float('learning_rate', 0.0, 'Initial learning rate.')
# define maximum iteration number
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
# a batch data number of samples assigned to each GPU
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './models'


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


def average_gradients(tower_grads):
    """
    Find the average number of updates for a given variable.

    :param tower_grads: Holds a list of all variables and their updates on the GPU.

    :return average_grads: Holds a list of all variables and their average number of updates on the GPU.
    """
    # a list of all variables on the GPU and their average updates
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tower_loss(name_scope, logit, labels):
    """
    According to the given set of network output and the corresponding ground truth,
    calculating the total loss, and carry out the visualization by TensorBoard.

    :param name_scope: Loss name prefix recorded in TensorBoard log.
    :param logit: A set of network outputs.
    :param labels: A set of ground truth.

    :return total_loss: A tensor for total loss.
    """
    cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit)
                  )
    tf.summary.scalar(
                  name_scope + '_cross_entropy',
                  cross_entropy_mean
                  )
    weight_decay_loss = tf.get_collection('weightdecay_losses')
    tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss) )

    # calculate the total loss for the current tower.
    total_loss = cross_entropy_mean + weight_decay_loss
    # visualize the mean of total_loss by TensorBoard
    tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss) )
    return total_loss


def tower_acc(logit, labels):
    """
    Calculating accuracy

    :param logit: A set of network outputs.
    :param labels: A set of ground truth.
    :return accuracy: Accuracy
    """
    correct_predict = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    return accuracy


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


def _variable_with_weight_decay(name, shape, wd):
    """
    Applying Xavier initialization method on the CPU to get the variable
    with the given name and shape, and its L2 regularization term is saved.

    :param name: Name.
    :param shape: Shape.
    :param wd: L2 regularization coefficient.

    :return var: The variable.
    """
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var


def run_training():
    """
    Get the sets of images and labels for training, validation, and
    tell TensorFlow that the model will be built into the default Graph.
    """
    # create directory for saving model
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # set the path to use the pre-trained model
    use_pretrained_model = True
    model_filename = "./C3D-tensorflow/sports1m_finetuning_ucf101.model"
    
    """
    with tf.Graph().as_default():
        # variable that sets the number of global iterations
        # (this variable increases by 1 after each iteration)
        global_step = tf.get_variable(
                        'global_step',
                        [],
                        initializer=tf.constant_initializer(0),
                        trainable=False
                        )
        # initialize a batch sample of the input model corresponding to the tensor.
        images_placeholder, labels_placeholder = placeholder_inputs(
                        FLAGS.batch_size * gpu_num
                        )
        tower_grads1 = []
        tower_grads2 = []
        logits = []
        # Adam optimization algorithm with learning rate of 0.0001 and 0.001
        # was adopted to optimize the feature extractor and classifier of the network
        # https://docs.w3cub.com/tensorflow~python/tf/train/adamoptimizer
        # https://arxiv.org/abs/1412.6980
        opt_stable = tf.train.AdamOptimizer(1e-4)
        opt_finetuning = tf.train.AdamOptimizer(1e-3)
        with tf.variable_scope('var_name') as var_scope:
            # based on the definition of C3D network structure
            # Learning Spatiotemporal Features with 3D Convolutional Networks
            # https://arxiv.org/pdf/1412.0767.pdf
            weights = {
                  'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
                  'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                  'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                  'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
                  'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                  'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
                  'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                  'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
                  'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
                  'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                  'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005)
                  }
            biases = {
                  'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
                  'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
                  'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
                  'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
                  'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
                  'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
                  'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
                  'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
                  'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
                  'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
                  'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000)
                  }
    """
    # variable that sets the number of global iterations
    # (this variable increases by 1 after each iteration)
    global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )
    # initialize a batch sample of the input model corresponding to the tensor.
    images_placeholder, labels_placeholder = placeholder_inputs(
                    FLAGS.batch_size * gpu_num
                    )
    tower_grads1 = []
    tower_grads2 = []
    logits = []
    # Adam optimization algorithm with learning rate of 0.0001 and 0.001
    # was adopted to optimize the feature extractor and classifier of the network
    # https://docs.w3cub.com/tensorflow~python/tf/train/adamoptimizer
    # https://arxiv.org/abs/1412.6980
    opt_stable = tf.train.AdamOptimizer(1e-4)
    opt_finetuning = tf.train.AdamOptimizer(1e-3)
    with tf.variable_scope('var_name') as var_scope:
        # based on the definition of C3D network structure
        # Learning Spatiotemporal Features with 3D Convolutional Networks
        # https://arxiv.org/pdf/1412.0767.pdf
        weights = {
              'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
              'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
              'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
              'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
              'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
              'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
              'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
              'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
              'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
              'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
              'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005)
              }
        biases = {
              'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
              'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
              'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
              'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
              'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
              'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
              'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
              'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
              'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
              'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
              'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000)
              }
    
    # traversing the GPU, distributing a batch data to each GPU for training,
    # and calculating the network output and average total loss.

    # dividing all parameters to be trained in the network into
    # feature extractor parameters and classifier parameters,
    # and applying different optimization algorithms to calculate
    # the updating amount of parameters and add them to the list.
    for gpu_index in range(0, gpu_num):
        with tf.device('/gpu:%d' % gpu_index):
            varlist2 = [weights['out'], biases['out'] ]
            # https://stackoverflow.com/questions/49312282/typeerror-unsupported-operand-types-for-dict-values-and-dict-values
            # varlist1 = list( set(weights.values() + biases.values()) - set(varlist2) )
            varlist1 = list((set(weights.values()) | set(biases.values())) - set(varlist2))
            logit = c3d_model.inference_c3d(
                            images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size, :, :, :, :],
                            0.5,
                            FLAGS.batch_size,
                            weights,
                            biases
                            )
            loss_name_scope = ('gpud_%d_loss' % gpu_index)
            loss = tower_loss(
                            loss_name_scope,
                            logit,
                            labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                            )
            grads1 = opt_stable.compute_gradients(loss, varlist1)
            grads2 = opt_finetuning.compute_gradients(loss, varlist2)
            tower_grads1.append(grads1)
            tower_grads2.append(grads2)
            logits.append(logit)
    # calculating accuracy and visualizing with TensorBoard
    logits = tf.concat(logits, 0)
    accuracy = tower_acc(logits, labels_placeholder)
    tf.summary.scalar('accuracy', accuracy)
    # calculating the average updating quantity of the parameters
    # of feature extractor and classifier, and applied respectively.
    grads1 = average_gradients(tower_grads1)
    grads2 = average_gradients(tower_grads2)
    apply_gradient_op1 = opt_stable.apply_gradients(grads1)
    apply_gradient_op2 = opt_finetuning.apply_gradients(grads2, global_step=global_step)
    # define an operation to update a variable with a moving average of MOVING_AVERAGE_DECAY,
    # variables in all the trainable parameters are updated each time while this performed.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    # based on Line 410-411 of moving_averages.py, 'tf.trainable_variables' can be removed
    # (C:\ProgramData\Anaconda3\lib\site-packages\tensorflow_core\python\training\moving_averages.py)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables)
    variables_averages_op = variable_averages.apply()
    # creates an action that groups all the operations that update the parameters.
    train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)
    null_op = tf.no_op()

    # create a variable to hold all weights and biases.
    # https://stackoverflow.com/questions/49312282/typeerror-unsupported-operand-types-for-dict-values-and-dict-values
    # saver = tf.train.Saver(weights.values() + biases.values())
    saver = tf.train.Saver(set(weights.values()) | set(biases.values()))
    # create an action that initializes all variables.
    init = tf.global_variables_initializer()

    # create a session that runs all the operations in the calculation diagram
    # and automatically selects the device to run on.
    sess = tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)
                    )
    # initialize all variables
    sess.run(init)
    # if the pretrained model exists and needs to be used,
    # the pretrained model is loaded.
    if os.path.isfile(model_filename) and use_pretrained_model:
        saver.restore(sess, model_filename)

    # merge the data together and write the log to the corresponding folder.
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./visual_logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/test', sess.graph)

    # randomly obtained a batch of training samples each time
    # for training and recording the iteration time,
    # the model is saved once for 10 iterations.

    # the test is carried out on the current training sample
    # and a randomly obtained batch test sample,
    # and the accuracy information is printed out,
    # while the TensorBoard log information is saved.
    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
                      filename='train.list',
                      batch_size=FLAGS.batch_size * gpu_num,
                      num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                      crop_size=c3d_model.CROP_SIZE,
                      shuffle=True
        )
        sess.run(train_op, feed_dict={
                      images_placeholder: train_images,
                      labels_placeholder: train_labels
                      }
                 )
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))

        # save a checkpoint and evaluate the model periodically.
        if step % 10 == 0 or (step + 1) == FLAGS.max_steps:
            saver.save(sess, os.path.join(model_save_dir, 'c3d_ucf_model'), global_step=step)
            print('Training Data Eval:')
            summary, acc = sess.run(
                            [merged, accuracy],
                            feed_dict={
                                images_placeholder: train_images,
                                labels_placeholder: train_labels
                                }
            )
            print("accuracy: " + "{:.5f}".format(acc))
            train_writer.add_summary(summary, step)
            print('Validation Data Eval:')
            val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
                            filename='test.list',
                            batch_size=FLAGS.batch_size * gpu_num,
                            num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                            crop_size=c3d_model.CROP_SIZE,
                            shuffle=True
                            )
            summary, acc = sess.run(
                            [merged, accuracy],
                            feed_dict={
                                images_placeholder: val_images,
                                labels_placeholder: val_labels
                                }
            )
            print("accuracy: " + "{:.5f}".format(acc))
            test_writer.add_summary(summary, step)
    print("done")


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
