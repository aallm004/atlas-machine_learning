#!/usr/bin/env python3
"""tensorflow for beginners"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """builds, trains, and saves a neural network classifier"""
    # Create the neural network
    def build_graph():
        """build graph function"""
        x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
        y_pred = forward_prop(x, layer_sizes, activations)
        loss = calculate_loss(y, y_pred)
        accuracy = calculate_accuracy(y, y_pred)
        train_op = create_train_op(loss, alpha)
        return x, y, y_pred, loss, accuracy, train_op

    def print_metrics(iteration, train_cost, train_accuracy, valid_cost, valid_accuracy):
        """print metrics function"""
        print(f"After {iteration} iterations:")
        print(f"\tTraining Cost: {train_cost}")
        print(f"\tTraining Accuracy: {train_accuracy}")
        print(f"\tValidation Cost: {valid_cost}")
        print(f"\tValidation Accuracy: {valid_accuracy}")


    # Call build_graph and unpack its return values
    x, y, y_pred, loss, accuracy, train_op = build_graph()

    # Add tensors to collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0  or i == iterations:
                print_metrics(i, train_cost, train_accuracy, valid_cost, valid_accuracy)

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        save_path = saver.save(sess, save_path)

    return save_path


