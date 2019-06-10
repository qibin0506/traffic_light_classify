import tensorflow as tf

import model
import utils
import images


def get_loss(result, y):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=result, labels=y)
    return tf.reduce_mean(cross_entropy)


def get_optimizer(loss):
    train_variables = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=utils.lr).minimize(loss, var_list=train_variables)
    return optimizer


def train():
    x_train, y_train, _ = images.read_traffic_light(True)
    x_test, y_test, _ = images.read_traffic_light(False)

    train_batches = x_train.shape[0]

    x, y, one_hot, result = model.get_model(is_train=True)
    loss = get_loss(result, one_hot)
    optimizer = get_optimizer(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(utils.epochs):
            for batch in range(train_batches // utils.batch_size):
                start = batch * utils.batch_size
                next_x = x_train[start:start + utils.batch_size]
                next_y = y_train[start:start + utils.batch_size]

                sess.run(optimizer, feed_dict={x: next_x, y: next_y})

            loss_result = sess.run(loss, feed_dict={x: x_test, y: y_test})
            print("epoch: {}, loss: {}".format(epoch, loss_result))

        saver.save(sess, "./result/result.ckpt")


if __name__ == '__main__':
    train()
