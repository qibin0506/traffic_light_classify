import tensorflow as tf
import numpy as np

import model
import images
import utils

import random


def test():
    x_test, y_test, raw_names = images.read_traffic_light(False)
    idxs = [random.randint(0, x_test.shape[0] - 1) for _ in range(200)]

    pics = []
    labels = []
    names = []

    for i in idxs:
        pics.append(x_test[i])
        labels.append(y_test[i])
        names.append(raw_names[i])

    x, _, _, result = model.get_model(is_train=False, keep_prob=1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./result/result.ckpt")

        dists = result.eval(feed_dict={x: pics})

        right_count = 0
        for i in range(len(dists)):
            print(i)
            dist = dists[i]
            pred_result = np.argmax(dist) == labels[i]
            if pred_result:
                right_count += 1

            print("{}: {} is {}, result is {}".format(pred_result, names[i],
                                                  utils.get_traffic_name(labels[i]),
                                                  utils.get_traffic_name(np.argmax(dist))))

        print("accuracy is {}".format(right_count / len(dists)))


if __name__ == '__main__':
    test()