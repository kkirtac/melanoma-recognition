import os,sys
sys.path.append("/valohai/inputs/models/models-master/research/slim")

import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow as tf

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
  #decoded_image = tf.image.resize_images(decoded_image, [224, 224])
  
  # input label is a grayscale image
  decoded_label = tf.image.decode_png(parsed["label"], channels=1)   
  #decoded_label = tf.image.resize_images(decoded_label, [224, 224])
  
  decoded_image = tf.to_float(decoded_image)
  
  # turn the label to floating-point with [0,1) range.
  decoded_label = tf.image.convert_image_dtype(decoded_label, tf.float32)
  decoded_label = tf.to_int32(decoded_label)
  
  # drop the last dimension to get shape (H,W) instead of (H,W,1)
  decoded_label = tf.squeeze(decoded_label, axis=[2])
  
  # repeat along the last dimension
  #decoded_label = tf.concat([decoded_label, 1-decoded_label], axis=2) 
  
  # per-channel mean values are taken from:
  # https://github.com/yulequan/melanoma-recognition/blob/master/segmentation/ResNet-50-f8s-skin-train-val.prototxt
  mean = tf.constant([182., 149., 135.],
                     dtype=tf.float32, shape=[1, 1, 3], name='img_mean')
  
  # center the image with per-channel RGB mean
  im_centered = decoded_image - mean
  #return (im_centered, decoded_label)
        
  # return features(image batch) as a dictionary containing min and max intensity values
  # return labels as a dictionary containing min and max intensity values
  return {"image": im_centered, 
          "min":tf.reduce_min(tf.reduce_min(im_centered, axis=1), axis=0),
          "max":tf.reduce_max(tf.reduce_max(im_centered, axis=1), axis=0)}, {"label": decoded_label, 
                                                                     "min":tf.reduce_min(tf.reduce_min(decoded_label, axis=1), axis=0),
                                                                     "max":tf.reduce_max(tf.reduce_max(decoded_label, axis=1), axis=0)}
      
      
def my_input_fn(filename, batch_size, epochs):
  """ reads the .tfrecords file,
  process data using parser method,
  shuffles data and returns the next batch.
  
  instantiates a tf.Dataset object and obtains a tf.data.Iterator, 
  that throws outOfRangeError when next batch goes out of the given epoch limit.
  more info: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
  """
  
  dataset = tf.data.TFRecordDataset(
      [filename]
  ).map(
      parser
  ).shuffle(
      buffer_size=100
  ).batch(
      batch_size
  ).repeat(
      epochs
  )
  
  # this returns the next batch from the parser function
  # the returned tuple is a dictionary for image and a dictionary for labels
  return dataset.make_one_shot_iterator().get_next() 
          
          
def get_deconv_filter(f_shape):
  """initialize a bilinear interpolation filter with
  given shape.
  
  this filter values are trained and updated in the current computation graph.
  """
  width = f_shape[0]
  heigh = f_shape[0]
  f = np.ceil(width/2.0)
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
  var = tf.get_variable(name="up_filter", initializer=init,
                        shape=weights.shape)
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
    num_input = ksize * ksize * in_features / stride
    stddev = (2 / num_input)**0.5
    weights = get_deconv_filter(f_shape)
    deconv = tf.nn.conv2d_transpose(x, weights, output_shape, strides = strides, padding='SAME')
    return deconv

def score_layer(x, name, num_classes, stddev = 0.001): 
  """receives a feature map and trains a linear scoring filter Wx+b
  
  this result with a new feature map with a consistent shape to be fused (per-pixel addition)
  with the corresponding upsampling result from the same input feature map x.
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
    # get number of input channels
    in_features = x.get_shape()[3].value
    shape = [1, 1, in_features, num_classes]
    w_decay = 5e-4
    init = tf.truncated_normal_initializer(stddev = stddev)
    weights = tf.get_variable("weights", shape = shape, initializer = init)
    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES

    if not tf.get_variable_scope().reuse:
      weight_decay = tf.multiply(tf.nn.l2_loss(weights), w_decay, name='weight_loss')
      tf.add_to_collection(collection_name, weight_decay)

    conv = tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding='SAME')
    
    # Apply bias
    initializer = tf.constant_initializer(0.0)
    conv_biases = tf.get_variable(name='biases', shape=[num_classes],initializer=initializer)
    bias = tf.nn.bias_add(conv, conv_biases)
    
    return bias     
      

def softmax_cross_entropy(logits, labels, num_classes):
  """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [N, H, W, num_classes].
          Use upscore32 as logits.
      labels: Labels tensor, int32 - [N, H, W, num_classes].
          The ground truth of your data.
    Returns:
      loss: Loss tensor of type float.
  """
  with tf.name_scope('softmax_ce_loss'):
    logits = tf.reshape(logits, (-1, num_classes))
    epsilon = tf.constant(value=1e-4)
    labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))
    softmax = tf.nn.softmax(logits) + epsilon
    cross_entropy = -tf.reduce_sum(labels * tf.log(softmax), 
                                   reduction_indices=[1])
    
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='xentropy_mean')
    
    _loss = cross_entropy_mean + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    
    tf.add_to_collection(tf.GraphKeys.LOSSES, _loss) 
    
    return _loss
  
  
def sigmoid_cross_entropy(logits, labels, num_classes):
  """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [N, H, W, num_classes].
          Use upscore32 as logits.
      labels: Labels tensor, int32 - [N, H, W, num_classes].
          The ground truth of your data.
    Returns:
      loss: Loss tensor of type float.
  """
  with tf.name_scope('sigmoid_ce_loss'):
    
    logits = tf.reshape(logits, (-1, num_classes))
    labels = tf.reshape(labels, (-1, num_classes))
    
    return tf.losses.sigmoid_cross_entropy(labels, logits)
  
  
def inference(features, labels, params):
  
  num_classes=params['num_classes']
  
  # output_stride denotes the rate of downsampling in the resnet network,i.e.,
  # 32 results with a feature map with a 1/32 downsampling rate such as,
  # (224,224) input is decreased to (8,8) resolution. 
  # Then, the rest of the code starts upsampling from this output, until reaching
  # the input resolution.
  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net, end_points = resnet_v1.resnet_v1_50(features["image"],
                                             global_pool=False, 
                                             output_stride=32)
    
    scale5 = net
    scale4 = end_points['resnet_v1_50/block3/unit_5/bottleneck_v1']
    scale3 = end_points['resnet_v1_50/block2/unit_3/bottleneck_v1']
    scale2 = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
    input_x = features["image"]
            
      # stride=32
      #resnet_v1_50/block3/unit_5/bottleneck_v1 (1, 15, 15, 1024)
      #net.shape: (1, 8, 8, 2048)  
      #resnet_v1_50/block2/unit_3/bottleneck_v1 (1, 29, 29, 512)
      #resnet_v1_50/block1/unit_2/bottleneck_v1 (1, 57, 57, 256)
      #resnet_v1_50/conv1 (1, 113, 113, 64)
      
  with tf.variable_scope('scale_fcn'):
    upscore2 = upscore_layer(scale5, shape = tf.shape(scale4), num_classes = num_classes, name = "upscore2", ksize = 4, stride = 2) 
    score_scale4 = score_layer(scale4, "score_scale4", num_classes = num_classes)
    fuse_scale4 = tf.add(upscore2, score_scale4)
      
    upscore4 = upscore_layer(fuse_scale4, shape = tf.shape(scale3), num_classes = num_classes, name = "upscore4", ksize = 4, stride = 2) 
    score_scale3 = score_layer(scale3, "score_scale3", num_classes = num_classes)
    fuse_scale3 = tf.add(upscore4, score_scale3)
      
    upscore8 = upscore_layer(fuse_scale3, shape = tf.shape(scale2), num_classes = num_classes, name = "upscore8", ksize = 4, stride = 2) 
    #score_scale2 = score_layer(scale2, "score_scale2", num_classes = num_classes)
    #fuse_scale2 = tf.add(upscore8, score_scale2)
    
    #upscore32 = upscore_layer(fuse_scale2, shape = tf.shape(input_x), num_classes = num_classes, name = "upscore32", ksize = 8, stride = 4)
      
    upscore32 = upscore_layer(upscore8, shape = tf.shape(input_x), num_classes = num_classes, name = "upscore32", ksize = 8, stride = 4)
    
    pred_up = tf.argmax(upscore32, axis = 3)
    pred = tf.expand_dims(pred_up, dim = 3, name='pred')     
      
  # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
  # We can then call the total loss easily    

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  print(sess.run(tf.shape(features["image"])))
  print(sess.run(tf.shape(upscore32)))
  print(sess.run(tf.shape(labels["label"])))
  sess.close()
    
  tf.losses.sparse_softmax_cross_entropy(labels=labels["label"], logits=upscore32)
  loss = tf.losses.get_total_loss()
  
  ## Compute evaluation metrics.
  # compute mean intersection over union, which corresponds to JA metric in the paper.
  accuracy = tf.metrics.mean_iou(labels=tf.reshape(tf.argmax(labels["label"],axis=2), [-1]),
                                 num_classes=2,
                                 predictions=tf.to_int32(tf.reshape(pred_up, [-1])),name="acc_op")
  
  
  ## SOFTMAX CROSS ENTROPY ##
  #loss = softmax_cross_entropy(logits=tf.to_float(upscore32), labels=labels["label"], num_classes=num_classes)

  
  ## SIGMOID CROSS ENTROPY  ##
  #loss = sigmoid_cross_entropy(logits=tf.to_float(upscore32), 
  #                             labels=labels["label"], 
  #                             num_classes=num_classes)

  #accuracy = tf.metrics.mean_iou(
   #   labels=tf.reshape(tf.argmax(labels["label"],axis=2), [-1]), 
    #  num_classes=2, 
     # predictions=tf.to_int32(tf.reshape(tf.argmax(upscore32,axis=2), [-1])), name="acc_op")
  ############################
  
    
  tf.summary.scalar('accuracy', accuracy[0])
  
  global_step = tf.train.get_or_create_global_step()     
      
  # set initial learning rate and 
  # scale it by 0.1 in every 3000 steps
  # as stated in https://github.com/yulequan/melanoma-recognition/blob/master/segmentation/solver.prototxt
  boundaries = [i*params['decay_steps'] for i in range(1, int(np.ceil(params['max_steps']/params['decay_steps']))) ]
  boundaries[-1] = params['max_steps']
  initial = [params['initial_learning_rate']]
  values = initial + [params['initial_learning_rate']*(0.1**i) for i in range(1,len(boundaries)+1)]
  lr = tf.train.piecewise_constant(global_step, boundaries, values)


  # Variables that affect learning rate
  var_resnet_batchnorm = [var for var in tf.trainable_variables() if ('conv1/BatchNorm' in var.name or 'conv2/BatchNorm' in var.name or 'conv3/BatchNorm' in var.name)] 
  
  var_upscale = [var for var in tf.trainable_variables() if 'score' in var.name]
  for varr in var_upscale:
    print(varr.name)
  
  var_rest = [var for var in tf.trainable_variables() if var.name not in var_resnet_batchnorm+var_upscale]
  
  # this is as stated in the paper and prototxt file,
  # https://github.com/yulequan/melanoma-recognition/blob/master/segmentation/ResNet-50-f8s-skin-train-val.prototxt
  # batchnorm filters placed just after each convolutional layer in each resnet block, are not trained.
  # so we set zero learning for these variables
  opt1 = tf.train.GradientDescentOptimizer(0)
  
  # this is also due to the prototxt file and the paper,
  # scoring and upsampling filters receive 0.1 * learning rate
  # https://github.com/yulequan/melanoma-recognition/blob/master/segmentation/ResNet-50-f8s-skin-train-val.prototxt
  opt2 = tf.train.GradientDescentOptimizer(lr*0.1)
  
  # rest of the variables receive the current learning rate
  opt3 = tf.train.GradientDescentOptimizer(lr)
  
  # gradient op: obtain the gradients with loss and given variables
  grads = tf.gradients(loss, var_resnet_batchnorm + var_upscale + var_rest)
  
  # grads for the first set of variables, currently has no effect due to zero learning rate.
  grads1 = grads[:len(var_resnet_batchnorm)]
  
  # grads for scoring and upsampling filters. Will get updated based on 0.1*learning_rate
  grads2 = grads[len(var_resnet_batchnorm):len(var_resnet_batchnorm)+len(var_upscale)]
  
  # grads for rest of the variables
  grads3 = grads[len(var_resnet_batchnorm)+len(var_upscale):]
  
  train_op1 = opt1.apply_gradients(zip(grads1, var_resnet_batchnorm), global_step=global_step)
  train_op2 = opt2.apply_gradients(zip(grads2, var_upscale), global_step=global_step)
  train_op3 = opt3.apply_gradients(zip(grads3, var_rest), global_step=global_step)
  
  train_op = tf.group(train_op1, train_op2, train_op3)


  ##### exponential weight decaying for learning rate #####
#  lr = tf.train.exponential_decay(params['initial_learning_rate'],
#                                 global_step,
#                                params['decay_steps'],
#                               params['learning_rate_decay_factor'],
#                              staircase=True)
#  
#  train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step) 
  ##### exponential weight decaying for learning rate #####
  
      
  return loss, accuracy[0], train_op
      

filename = '/valohai/inputs/training-data/melanoma_train.tfrecords'
batch_size=1
epochs=50
save_model_path='/valohai/outputs'
restore_ckpt_path='/valohai/inputs/pretrained/resnet_v1_50.ckpt'
initial_learning_rate=1e-3
num_classes=2
params = {'initial_learning_rate' : initial_learning_rate,
          'learning_rate_decay_factor' : 0.96,
          'num_classes' : num_classes,
          'decay_steps' : 3000,
          'max_steps' : 100000}
  
  
features_op, labels_op = my_input_fn(filename, batch_size, epochs) 
ops = inference(features_op, labels_op, params)
  
  
with tf.Session() as sess:
  
  exclude_restore = [var.name for var in tf.global_variables() if ('logits' in var.name or 'scale_fcn' in var.name) ] 
  
  
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  
  variables_to_restore = slim.get_variables_to_restore(exclude=exclude_restore)
  tf.train.init_from_checkpoint(restore_ckpt_path,
                                {v.name.split(':')[0]: v for v in variables_to_restore})
  
  print('RESTORE COMPLETE!')
  
  saver = tf.train.Saver()
  
  train_step = 0
  print_every = 100
  while True:
    try:
      
      #print(sess.run(labels_op))
      
      loss, acc, _ = sess.run(ops)

      
      train_step += 1
      
      print('training loss: {},  step: {}'.format(loss, train_step))
      
      if train_step % print_every == 0:
        print('training mIOU: {},  step: {}'.format(acc, train_step))

      if train_step % 10000 == 0:
        save_path = saver.save(sess, save_model_path)

      
      
    except tf.errors.OutOfRangeError:
      
      print('Training finished. Saving the resulting model to {}'.format(save_model_path))
      save_path = saver.save(sess, save_model_path)
      
      break
