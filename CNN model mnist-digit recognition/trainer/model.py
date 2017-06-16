# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import metrics
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


tf.logging.set_verbosity(tf.logging.INFO)

DEBUG =0 

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
      serialized_example,
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape([784])
  image = tf.cast(image, tf.float32) * (1. / 255)
  label = tf.cast(features['label'], tf.int32)

  return image, label


def input_fn(filename, batch_size=100, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      [filename], num_epochs=num_epochs)

  image, label = read_and_decode(filename_queue)
  images, labels = tf.train.batch(
      [image, label], batch_size=batch_size,
      capacity=1000 + 3 * batch_size)

  return {'image': images}, labels


def get_input_fn(filename, num_epochs=None, batch_size=100):
  return lambda: input_fn(filename, batch_size)


def _cnn_model_fn(features, labels, mode):
  # Input Layer
  # input_layer = tf.reshape(features['image'], [-1, 28, 28, 1])
  X= tf.reshape(features['image'], [-1, 28, 28, 1])
  # print ("input layer: ", input_layer)

  ## try to emulate mnist program mnist_3.1_convolutional_bigger_dropout.py
   
  # Convolutional Layer #1
  conv0 = tf.layers.conv2d(
      inputs=X,
      filters=16,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  if DEBUG:  print ("convolution layer0: ", conv0) 
 
  
  # Pooling Layer #0
  pool0 = tf.layers.max_pooling2d(inputs=conv0, pool_size=[2, 2], strides=1)
  
  if DEBUG: print ("shape pool0: " + str(tf.shape(pool0) ) )
  if DEBUG: print ("pool0: ", pool0) 
  

  # Convolutional Layer #2
  conv1 = tf.layers.conv2d(
      inputs=pool0,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  if DEBUG: print ("conv1: ", conv1) 
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  if DEBUG: print ("pool1: ", pool1) 

  # Convolutional Layer #3 and Pooling Layer #3
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  
  if DEBUG: print ("conv2: ", conv2) 

  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  if DEBUG: print ("pool2: ", pool2) 

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 6  * 6 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=(mode == learn.ModeKeys.TRAIN))

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)
 
  if DEBUG: print ("shape of logits: " + str( tf.shape(logits)) )

  '''
  # three convolutional layers with their channel counts, and a
  # fully connected layer (tha last layer has 10 softmax neurons)
  K = 6  # first convolutional layer output depth
  L = 12  # second convolutional layer output depth
  M = 24  # third convolutional layer
  N = 200  # fully connected layer

  W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1),name="w1_var" )  # 6x6 patch, 1 input channel, K output channels
  B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]), name="w1_var")
  W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1), name="w2_var")
  B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]),name="b2_var")
  W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1),name="w3_var")
  B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]),name="b3_var")

  W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1),name="w4_var")
  B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]),name="b4_var")
  W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1),name="w5_var")
  B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]),name="b5_var")
 
  # bility of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
  pkeep = tf.placeholder(tf.float32)
  pkeep = 0.75


  # The model
  stride = 1  # output is 28x28
  Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
  stride = 2  # output is 14x14
  Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
  stride = 2  # output is 7x7
  Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

  # reshape the output from the third convolution for the fully connected layer
  YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

  Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
  YY4 = tf.nn.dropout(Y4, pkeep)
  Ylogits = tf.matmul(YY4, W5) + B5
  logits = tf.nn.softmax(Ylogits)

  '''

  loss = None
  train_op = None
 
  if DEBUG: print("logits size: ", tf.shape(logits)) 


  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    if DEBUG: print("onehot_labels:" + str(tf.shape(onehot_labels)  ))
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001, optimizer="Adam")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  print ("finished modelling of input")
  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(mode=mode, loss=loss, train_op=train_op,
                                 predictions=predictions)


def build_estimator(model_dir):
  return learn.Estimator(
           model_fn=_cnn_model_fn,
           model_dir=model_dir,
           config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))


def get_eval_metrics():
  return {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,
                                       prediction_key="classes")
  }


def serving_input_fn():
  feature_placeholders = {'image': tf.placeholder(tf.float32, [None, 784])}
  features = {
    key: tensor
    for key, tensor in feature_placeholders.items()
  }    
  return learn.utils.input_fn_utils.InputFnOps(
    features,
    None,
    feature_placeholders
  )
