"""
VGG transfer learning Test
"""
import tensorflow as tf
import myVGG as myvgg19
import utils
import numpy as np
from sklearn.utils import shuffle
import cifar10.cifar10 as data
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt


#----------------load CIFAR10-------------------#
class_names = data.load_class_names()
images_train, cls_train, labels_train = data.load_training_data()
images_test, cls_test, labels_test = data.load_test_data()

# print(len(images_test))


#-----------------------resize-------------------#

batch = imresize(images_test[0], (224, 224))


#see how resize effect
# plt.imshow(batch)
# plt.show()
# plt.imshow(batch)
# plt.show()

#----------------set up---------------------#
batch_size = 1
learning_rate = 0.0001
images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
train_mode = tf.placeholder(tf.bool)
print(train_mode)

vgg = myvgg19.Vgg19(vgg19_npy_path='./vgg19.npy', retrain=['fc8'], outputclass=10)
vgg.build(images, train_mode)

# print number of variables used: 143667240 variables, i.e. ideal size = 548MB
print(vgg.get_var_count())


merged_summary_op = tf.summary.merge_all()


#-------------------------------start training----------------------------------------#
# a = False
a = True
if a:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # test classification
    feature = sess.run(vgg.pool1, feed_dict={images: [batch], train_mode: False})[0]
    fm = np.transpose(feature, axes=[2, 0, 1])

    #show
    plt.show()
    for m in fm:
        plt.imshow(m)
        plt.pause(0.1)
