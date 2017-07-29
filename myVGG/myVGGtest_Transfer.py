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

train_input = []
test_input = []
#print(len(images_test))


#-----------------------resize-------------------#
print("resize start")
for i in range(len(images_test)):
        tmp = imresize(images_test[i], (224, 224))
        test_input.append(tmp)

print("resize complete")

#see how resize effect
# plt.imshow(images_test[0])
# plt.show()
# plt.imshow(test_input[0])
# plt.show()

#----------------model set up---------------------#
batch_size = 8
learning_rate = 0.0001
train_layers1 = ['fc8', 'fc7', 'fc6']
images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [batch_size, 10])
train_mode = tf.placeholder(tf.bool)
print(train_mode)

vgg = myvgg19.Vgg19(vgg19_npy_path='./vgg19.npy', retrain=['fc8'], outputclass=10)
vgg.build(images, train_mode)

# print number of variables used: 143667240 variables, i.e. ideal size = 548MB
print(vgg.get_var_count())


#--------------training set up -----------------#
with tf.name_scope("cost_function") as scope:
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    tf.summary.scalar("loss", cost)

with tf.name_scope("train") as scope:
    var_list1 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers1]
    print(var_list1)
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, var_list=var_list1)

merged_summary_op = tf.summary.merge_all()


#-------------------------------start training----------------------------------------#
# a = False
a = True
if a:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("logs/", sess.graph)

    print("training start")
    for i in range(1000):
        print('training step ', i)
        for j in range(0,len(test_input),batch_size):
            sess.run(train, feed_dict={images: test_input[j: j + batch_size], true_out: labels_test[j: j + batch_size], train_mode: True})

        # Write logs for each iteration
        summary_str = sess.run(merged_summary_op, feed_dict={images: test_input[j: j + batch_size],
                                                            true_out: labels_test[j: j + batch_size],
                                                            train_mode: True})
        writer.add_summary(summary_str, i * (len(test_input) / batch_size) + j)
        #re-shuffle
        print("reshuffle")
        test_input, labels_test = shuffle(test_input, labels_test)
        print("reshuffle done")

    print('training done')

    #see the result
    # result = []
    # # for i in range(0, len(test_input), batch_size):
    # #     prob = sess.run(vgg.prob, feed_dict={images: test_input[i: i + batch_size], train_mode: False})
    # #     result.append(prob)
    # # print(result)
    # # print(labels_test)

    print('start Evaluation')
    correct_prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(true_out, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #accuracy
    for j in range(0, len(test_input), batch_size):
        print "Accuracy:", accuracy.eval(session=sess, feed_dict={images: test_input[j: j + batch_size], true_out: labels_test[j: j + batch_size], train_mode: False})


    # test save
    #vgg.save_npy(sess, './test-save.npy')
