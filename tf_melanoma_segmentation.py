import os, sys

sys.path.append("/valohai/inputs/models/models-master/research/slim")

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import datetime, os, glob, errno
from tensorflow.python.platform import tf_logging as logging


def parser(record):
    """Parses input records and returns batch_size samples
    """

    # each tf.Example object in input .tfrecords file packs image and label as,
    # dictionary with key: 'image', its value: a png-encoded bytearray representation
    # dictionary with key: 'label', its value: a png-encoded bytearray representation,
    # label is a binary image with 255 pixel value for foreground, 0 for background.
    keys_to_features = {
        "image": tf.FixedLenFeature((), tf.string, default_value=""),
        "label": tf.FixedLenFeature((), tf.string, default_value="")
    }

    parsed = tf.parse_single_example(record, keys_to_features)

    # input image is a 3-channel RGB image
    decoded_image = tf.image.decode_png(parsed["image"], channels=3)

    # input label is a grayscale image
    decoded_label = tf.image.decode_png(parsed["label"], channels=1)

    # turn the label to floating-point with [0,1) range.
    decoded_label = tf.image.convert_image_dtype(decoded_label, tf.float32)
    # decoded_label = tf.round(decoded_label)
    decoded_label = tf.to_int32(decoded_label)

    # drop the last dimension to get shape (H,W) instead of (H,W,1)
    # decoded_label = tf.squeeze(decoded_label, axis=[2])

    # repeat along the last dimension
    decoded_label = tf.concat([decoded_label, 1 - decoded_label], axis=2)
    ch0, ch1 = tf.split(value=decoded_label, num_or_size_splits=2, axis=2)
    decoded_label = tf.concat(values=[ch1, ch0], axis=2)

    # per-channel mean values are taken from:
    # https://github.com/yulequan/melanoma-recognition/blob/master/segmentation/ResNet-50-f8s-skin-train-val.prototxt
    mean = tf.constant([184., 153., 138.],
                       dtype=tf.float32, shape=[1, 1, 3], name='img_mean')

    # center the image with per-channel RGB mean
    decoded_image = tf.to_float(decoded_image)
    im_centered = decoded_image - mean
    # return (im_centered, decoded_label)

    # im_centered = tf.image.per_image_standardization(decoded_image)

    imagedict = {'image': im_centered}
    labeldict = {'label': decoded_label}

    return im_centered, decoded_label


def preproc_train(sample, target):
    seed = np.random.randint(0, 2 ** 32)

    cropsize_img = [480, 480, 3]
    cropsize_label = [480, 480, 2]

    image = tf.random_crop(sample, cropsize_img, seed=seed, name='crop_mirror_train_img')
    label = tf.random_crop(target, cropsize_label, seed=seed, name='crop_mirror_train_label')

    tf.summary.image('crop_mirror_train_img', image)
    tf.summary.image('crop_mirror_train_label', tf.cast(label, dtype=tf.uint8))

    image = tf.image.random_flip_left_right(image, seed=seed)
    label = tf.image.random_flip_left_right(label, seed=seed)

    return image, label


def preproc_val(sample, target):
    seed = np.random.randint(0, 2 ** 32)

    image = tf.image.resize_image_with_crop_or_pad(sample, 480, 480)
    label = tf.image.resize_image_with_crop_or_pad(target, 480, 480)

    image = tf.image.random_flip_left_right(image, seed=seed)
    label = tf.image.random_flip_left_right(label, seed=seed)

    return image, label


def my_input_fn_train(filename, batch_size, epochs):
    """ reads the .tfrecords file,
    process data using parser method,
    shuffles data and returns the next batch.

    instantiates a tf.Dataset object and obtains a tf.data.Iterator,
    that throws outOfRangeError when next batch goes out of the given epoch limit.
    more info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    """

    dataset = tf.data.TFRecordDataset([filename])
    augmented = dataset.map(parser
                            ).map(
        preproc_train
    ).shuffle(
        buffer_size=1000
    ).batch(
        batch_size
    ).repeat(
        epochs
    )

    # this returns the next batch from the parser function
    # the returned tuple is a dictionary for image and a dictionary for labels
    features, labels = augmented.make_one_shot_iterator().get_next()
    return {'image': features}, {'label': labels}


def my_input_fn_val(filename, batch_size, epochs):
    """ reads the .tfrecords file,
    process data using parser method,
    shuffles data and returns the next batch.

    instantiates a tf.Dataset object and obtains a tf.data.Iterator,
    that throws outOfRangeError when next batch goes out of the given epoch limit.
    more info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    """

    dataset = tf.data.TFRecordDataset([filename])
    augmented = dataset.map(parser
                            ).map(
        preproc_val
    ).batch(
        batch_size
    ).repeat(
        epochs
    )

    # this returns the next batch from the parser function
    # the returned tuple is a dictionary for image and a dictionary for labels
    features, labels = augmented.make_one_shot_iterator().get_next()
    return {'image': features}, {'label': labels}


def get_deconv_filter(f_shape):
    """initialize a bilinear interpolation filter with
    given shape.

    this filter values are trained and updated in the current computation graph.
    """
    width = f_shape[0]
    heigh = f_shape[0]
    f = np.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="up_filter",
                          initializer=init,
                          shape=weights.shape,
                          regularizer=tf.contrib.layers.l2_regularizer(5e-4))
    return var


def upscore_layer(x, shape, num_classes, name, ksize, stride):
    """transposed convolution filter to learn upsampling filter values.
    Given a feature map x, initialize a bilinear interpolation filter with
    given shape to upsample the map based on the given stride.

    this filter values are trained and updated in the current computation graph.
    """
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        in_features = x.get_shape()[3].value
        if shape is None:
            in_shape = tf.shape(x)
            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]

        output_shape = tf.stack(new_shape)
        f_shape = [ksize, ksize, num_classes, in_features]

        weights = get_deconv_filter(f_shape)

        # grp1 = tf.nn.conv2d_transpose(x[:, :, :, :int(in_features/2)], weights[:, :, 0, :], tf.stack([shape[0], shape[1], shape[2], 1]), strides = strides, padding='SAME')
        # grp2 = tf.nn.conv2d_transpose(x[:, :, :, int(in_features/2):], weights[:, :, 1, :], tf.stack([shape[0], shape[1], shape[2], 1]), strides = strides, padding='SAME')

        # deconv = tf.concat(axis=3, values=[grp1, grp2])

        deconv = tf.nn.conv2d_transpose(x, weights, output_shape, strides=strides, padding='SAME')

        return deconv


def score_layer(x, name, num_classes, stddev=0.001):
    """receives a feature map and trains a linear scoring filter Wx+b

    this result with a new feature map with a consistent shape to be fused (per-pixel addition)
    with the corresponding upsampling result from the same input feature map x.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # get number of input channels
        in_features = x.get_shape()[3].value
        shape = [1, 1, in_features, num_classes]
        w_decay = 5e-4
        # init = tf.truncated_normal_initializer(stddev = stddev)
        init = tf.constant_initializer(0.0)
        weights = tf.get_variable("weights",
                                  shape=shape,
                                  initializer=init,
                                  regularizer=tf.contrib.layers.l2_regularizer(w_decay))
        # collection_name = tf.GraphKeys.REGULARIZATION_LOSSES

        # if not tf.get_variable_scope().reuse:
        #  weight_decay = tf.multiply(tf.nn.l2_loss(weights), w_decay, name='weight_loss')
        #  tf.add_to_collection(collection_name, weight_decay)

        conv = tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding='SAME')

        # Apply bias
        initializer = tf.constant_initializer(0.0)
        conv_biases = tf.get_variable(name='biases', shape=[num_classes], initializer=initializer)
        bias = tf.nn.bias_add(conv, conv_biases)

        return bias


def resnet_v1_50_fcn(features, labels, mode, params):
    num_classes = params['num_classes']

    # output_stride denotes the rate of downsampling in the resnet network,i.e.,
    # 32 results with a feature map with a 1/32 downsampling rate such as,
    # (224,224) input is decreased to (8,8) resolution.
    # Then, the rest of the code starts upsampling from this output, until reaching
    # the input resolution.
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.0005)):
        net, end_points = resnet_v1.resnet_v1_50(features['image'],
                                                 global_pool=False,
                                                 output_stride=32)

        scale5 = net
        scale4 = end_points['resnet_v1_50/block3/unit_5/bottleneck_v1']
        scale3 = end_points['resnet_v1_50/block2/unit_3/bottleneck_v1']
        scale2 = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
        input_x = features['image']

    with tf.variable_scope('scale_fcn'):

        score_scale5 = score_layer(scale5, "score_scale5", num_classes=num_classes)
        upscore2 = upscore_layer(score_scale5, shape=tf.shape(scale4), num_classes=num_classes, name="upscore2",
                                 ksize=4, stride=2)

        score_scale4 = score_layer(scale4, "score_scale4", num_classes=num_classes)
        fuse_scale4 = tf.add(upscore2, score_scale4)

        upscore4 = upscore_layer(fuse_scale4, shape=tf.shape(scale3), num_classes=num_classes, name="upscore4", ksize=4,
                                 stride=2)
        score_scale3 = score_layer(scale3, "score_scale3", num_classes=num_classes)
        fuse_scale3 = tf.add(upscore4, score_scale3)

        upscore32 = upscore_layer(fuse_scale3, shape=tf.shape(input_x), num_classes=num_classes, name="upscore32",
                                  ksize=16, stride=8)

        pred_up = tf.argmax(upscore32, axis=3)  # shape: [4 480 480]
        pred = tf.expand_dims(pred_up, dim=3, name='pred')  # shape: [4 480 480 1]

    #       sess = tf.Session()
    #       sess.run(tf.global_variables_initializer())
    #       sess.run(tf.local_variables_initializer())
    #       print(sess.run(tf.shape(scale5)))
    #       print(sess.run(tf.shape(upscore32)))
    #       print(sess.run(tf.shape(pred_up)))
    #       print(sess.run(tf.shape(pred)))
    #       sess.close()

    tf.summary.image('ground-truth', tf.expand_dims(255.0 * tf.to_float(tf.argmax(labels['label'], axis=3)), axis=3))
    tf.summary.image('prediction', 255.0 * tf.to_float(pred))

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_up)

    # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
    # We can then call the total loss easily
    tf.losses.softmax_cross_entropy(onehot_labels=labels['label'],
                                    logits=upscore32)
    loss = tf.losses.get_total_loss()
    tf.summary.scalar('loss', loss)

    ## Compute evaluation metrics.
    # compute mean intersection over union, which corresponds to JA metric in the paper.
    iou_op, update_op = tf.metrics.mean_iou(
        labels=tf.reshape(tf.argmax(labels['label'], axis=3), [-1]),
        num_classes=2,
        predictions=tf.to_int32(tf.reshape(pred_up, [-1])), name="acc_op")

    with tf.control_dependencies([update_op]):
        iou = tf.identity(iou_op)

    tf.summary.scalar('iou', iou)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops={'iou': (iou_op, update_op)})

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    global_step = tf.train.get_or_create_global_step()

    training_loss = tf.identity(loss, name='training_loss')

    # set initial learning rate and
    # scale it by 0.1 in every 3000 steps
    # as stated in https://github.com/yulequan/melanoma-recognition/blob/master/segmentation/solver.prototxt
    boundaries = [i * params['decay_steps'] for i in
                  range(1, int(np.ceil(params['max_steps'] / params['decay_steps'])))]
    boundaries[-1] = params['max_steps']
    initial = [params['initial_learning_rate']]
    values = initial + [params['initial_learning_rate'] * (0.1 ** i) for i in range(1, len(boundaries) + 1)]
    lr = tf.train.piecewise_constant(global_step, boundaries, values)
    # lr = tf.constant(params['initial_learning_rate'])

    tf.summary.scalar('lr', lr)

    # Variables that affect learning rate
    var_resnet_batchnorm = [var for var in tf.trainable_variables()
                            if ('conv1/BatchNorm' in var.name or
                                'conv2/BatchNorm' in var.name or
                                'conv3/BatchNorm' in var.name or
                                'shortcut/BatchNorm' in var.name)]

    var_upscale = [var for var in tf.trainable_variables()
                   if 'score' in var.name and 'bias' not in var.name]

    var_score_bias = [var for var in tf.trainable_variables()
                      if 'score' in var.name and 'bias' in var.name]

    var_rest = [var for var in tf.trainable_variables()
                if var not in var_resnet_batchnorm + var_upscale + var_score_bias]

    # this is as stated in the paper and prototxt file,
    # https://github.com/yulequan/melanoma-recognition/blob/master/segmentation/ResNet-50-f8s-skin-train-val.prototxt
    # batchnorm variables placed just after each convolutional layer in each resnet block, are not updated.
    # so we set zero learning for these variables
    # opt1 = tf.train.MomentumOptimizer(0, 0.9)

    # this is also due to the prototxt file and the paper,
    # scoring and upsampling filters receive 0.1 * learning rate
    # https://github.com/yulequan/melanoma-recognition/blob/master/segmentation/ResNet-50-f8s-skin-train-val.prototxt
    opt_upscale = tf.train.MomentumOptimizer(lr * 0.1, 0.9)

    # rest of the variables receive the current learning rate
    opt_scorebias = tf.train.MomentumOptimizer(lr * 0.2, 0.9)

    # rest of the variables receive the current learning rate
    opt_rest = tf.train.MomentumOptimizer(lr, 0.9)

    # gradient op: obtain the gradients with loss and given variables
    grads = tf.gradients(loss, var_upscale + var_score_bias + var_rest)

    # grads for the first set of variables, currently has no effect due to zero learning rate.
    # grads1 = grads[:len(var_resnet_batchnorm)]

    # grads for scoring and upsampling filters. Will get updated based on 0.1*learning_rate
    grads_upscale = grads[:len(var_upscale)]

    # for i,val in enumerate(grads2):
    #  tf.summary.histogram('upscale_grads_{}'.format(i), val)

    # grads for bias variables
    grads_scorebias = grads[len(var_upscale):len(var_upscale) + len(var_score_bias)]

    # grads for the rest of variables
    grads_rest = grads[len(var_upscale) + len(var_score_bias):]

    # train_op1 = opt1.apply_gradients(zip(grads1, var_resnet_batchnorm), global_step=global_step)
    train_op_upscale = opt_upscale.apply_gradients(zip(grads_upscale, var_upscale), global_step=global_step)
    train_op_scorebias = opt_scorebias.apply_gradients(zip(grads_scorebias, var_score_bias), global_step=global_step)
    train_op_rest = opt_rest.apply_gradients(zip(grads_rest, var_rest), global_step=global_step)

    train_op = tf.group(train_op_upscale, train_op_scorebias, train_op_rest)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class MyLoggingAverageLossHook(tf.train.LoggingTensorHook):
    def __init__(self, tensors, every_n_iter):
        super().__init__(tensors=tensors, every_n_iter=every_n_iter)

        # keep track of previous losses
        self.losses = []
        self.every_n_iter = every_n_iter

    def after_run(self, run_context, run_values):
        _ = run_context

        self._tag = ''

        # please put only one loss tensor
        for tag in self._tag_order:
            self.losses.append(run_values.results[tag])
            self._tag = tag

        if self._should_trigger:
            self._log_tensors(run_values.results)

        self._iter_count += 1

    def _log_tensors(self, tensor_values):

        if self._iter_count % self.every_n_iter == 0:
            original = np.get_printoptions()
            np.set_printoptions(suppress=True)
            logging.info("%s = %s" % (self._tag, np.mean(self.losses)))
            np.set_printoptions(**original)
            self.losses = []


tf.reset_default_graph()

# filename_train = 'melanoma_train_224.tfrecords'
# filename_val = 'melanoma_val_224.tfrecords'
filename_train = '/valohai/inputs/training-data/melanoma_train_v8.tfrecords'
filename_val = '/valohai/inputs/validation-data/melanoma_val_v8.tfrecords'
save_model_dir = '/valohai/outputs'
restore_ckpt_path = '/valohai/inputs/pretrained/resnet_v1_50.ckpt'
save_checkpoints_steps = 3000
params = {'initial_learning_rate': 1e-3,
          'num_classes': 2,
          'decay_steps': 3000,
          'max_steps': 100000
          }

# with tf.Session() as sess:

# exclude_restore = [var.name for var in tf.global_variables() if not ('logits' in var.name or 'scale_fcn' in var.name or 'Momentum' in var.name) ]
# variables_to_restore = slim.get_variables_to_restore(exclude=exclude_restore)
# tf.train.init_from_checkpoint(restore_ckpt_path,
#                                {v.name.split(':')[0]: v for v in variables_to_restore})

tf.logging.set_verbosity(tf.logging.INFO)

run_config = tf.estimator.RunConfig(save_checkpoints_steps=save_checkpoints_steps)

avg_train_loss_log = {"average_training_loss": "training_loss"}

# this custom logging hook is created to average last every_n_iter loss values
# and display the result as an INFO
my_averageloss_logging_hook = MyLoggingAverageLossHook(
    tensors=avg_train_loss_log,
    every_n_iter=10)

ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=restore_ckpt_path,
                                    vars_to_warm_start='.*resnet_v1.*')

estimator = tf.estimator.Estimator(
    model_fn=resnet_v1_50_fcn,
    model_dir=save_model_dir,
    params=params,
    config=run_config,
    warm_start_from=ws
)

print(estimator.eval_dir())

print(os.listdir(save_model_dir))

# train_spec = tf.estimator.TrainSpec(input_fn=lambda:my_input_fn(filename_train, batch_size=4, epochs=500),
#                                       max_steps=params['max_steps'],
#                                       hooks=[my_averageloss_logging_hook])

train_spec = tf.estimator.TrainSpec(input_fn=lambda: my_input_fn_train(filename_train, batch_size=4, epochs=500),
                                    max_steps=params['max_steps'])

eval_spec = tf.estimator.EvalSpec(input_fn=lambda: my_input_fn_val(filename_val, batch_size=4, epochs=1),
                                  steps=None, throttle_secs=50)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def evaluate(filename_test, save_model_dir):
    tf.reset_default_graph()

    # filename_test = '/valohai/inputs/test-data/melanoma_test_v8.tfrecords'

    # save_model_dir='/valohai/outputs'
    params = {'initial_learning_rate': 1e-3,
              'num_classes': 2,
              'decay_steps': 3000,
              'max_steps': 100000
              }

    tf.logging.set_verbosity(tf.logging.INFO)

    estimator = tf.estimator.Estimator(
        model_fn=resnet_v1_50_fcn,
        model_dir=save_model_dir,
        params=params)

    estimator.evaluate(input_fn=lambda: my_input_fn_val(filename_test, batch_size=4, epochs=1))
