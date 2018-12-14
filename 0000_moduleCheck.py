import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import scipy
import sklearn
from tensorflow.python.client import device_lib


print("matplotlib version=", matplotlib.__version__)
print("opencv version=", cv2.__version__)
print("numpy version=", np.version.version)
print("tensorflow version=", tf.__version__)
print("scipy version=", scipy.__version__)
print("skit-learn version=", sklearn.__version__)
print("GPU check:", device_lib.list_local_devices())
if tf.test.is_built_with_cuda():
    print("The installed version of TensorFlow includes GPU support.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    print("The installed version of TensorFlow does not include GPU support.")


with tf.device('/GPU:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
print(sess.run(c))