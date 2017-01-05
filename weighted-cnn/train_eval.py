#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import random
import data_helpers
from random import randint
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from text_cnn import TextCNN
from tensorflow.contrib import learn


class Classifier(object):
    def __init__(self, checkpoint=None, accuracy=None, iteration=None):
        self.checkpoint = checkpoint
        self.accuracy = accuracy
        self.iteration = iteration
# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
random_seed = 10
run_number = 3
number_of_classifiers = 2

x_text, y = data_helpers.load_data_and_labels(1000,"negative")

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(random_seed)
shuffle_indices = np.random.permutation(np.arange(len(y)))

#x_raw_shuffled = x_text[shuffle_indices]
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]

np_dev_x = x_dev
np_dev_y = y_dev
print("Writing out x dev")
print(x_dev)
np.save('dev_x.txt',x_dev)
print("Writing out y dev")
print(y_dev)
np.save('dev_y.txt',y_dev)
list_minority_negative = []
classifier_list = []


for p in range(0,number_of_classifiers):
    rand_seed = randint(0, 9)
    #Get count of positive negative in train set
    pos_value = np.array([0,1])
    neg_value = np.array([1,0])

    list_positive_instances = []
    list_negative_instances = []
    list_positive_balanced = []
    list_negative_balanced = []

    for x in range(0, len(x_train)):
        if (y_train[x]==pos_value).all():
           #print("Positive Label")
           list_positive_instances.append(x_train[x])
        else:
           #print("Negative label")
           list_negative_instances.append(x_train[x])



    print("The count of positive labels in test: %s",len(list_positive_instances))
    print("The count of negative labels in test: %s",len(list_negative_instances))



    print("Undersampling the positive instances")
    for x in range(0,len(list_negative_instances)):
        list_positive_balanced.append(list_positive_instances[random.randint(0,len(list_positive_instances)-1)])
    print("Positive size now: {}".format(len(list_negative_balanced)))
    list_negative_instances = list_negative_instances
    list_positive_instances = list_positive_balanced

    #Regenerate the labels

    positive_labels = [[0,1] for _ in list_positive_instances]
    negative_labels = [[1,0] for _ in list_negative_instances]
    print("Length of positive labels:%s",len(positive_labels))
    print("Length of negative labels:%s",len(negative_labels))


    y_t = np.concatenate([positive_labels, negative_labels],0)
    x_t = np.array(list_positive_instances + list_negative_instances)

    np.random.seed(rand_seed)
    shuffle_indices = np.random.permutation(np.arange(len(y_t)))
    x_train = x_t[shuffle_indices]
    y_train = y_t[shuffle_indices]
    print("Overall Length:%s", len(y_train))


    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    run_accuracy = []

    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=2,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            print("Checkpoint Dir is {}".format(out_dir))

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                run_accuracy.append(accuracy)
                print("Run Accuracy List:")
                print(run_accuracy)
                print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #if writer:
            #        writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            classifier_list.append(Classifier(checkpoint=checkpoint_dir,accuracy=run_accuracy[len(run_accuracy)-1],iteration=p))
            print("The Final Accuracy is {}".format(run_accuracy))


#Print Classifier List and Sort
for t in classifier_list:
    print("\n {}  {} iteration {}".format(t.checkpoint, t.accuracy, t.iteration))
classifier_list.sort(key=lambda x: x.accuracy, reverse=True)
print("\nSelected Classifier {} has accuracy {}".format(classifier_list[0].checkpoint, classifier_list[0].accuracy))



#Begin Evaluation of Dev
x_raw = np.load("dev_x.txt.npy")
y_test = np.load("dev_y.txt.npy")
x_raw = np_dev_x
y_test = np_dev_y
print ("X TEST\n")
print(x_raw)
print ("End X RAW\n")
# y_test = np.argmax(y_test, axis=1)
#
# Map data into vocabulary
vocab_path = os.path.join(classifier_list[0].checkpoint, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.load("dev_x.txt.npy")
x_test = np_dev_x
#
# print("\nEvaluating...\n")
#
# # Evaluation
# # ==================================================
checkpoint_file = tf.train.latest_checkpoint(classifier_list[0].checkpoint)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
#
#         # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
#
#         # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
#
#         # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
#
#         # Collect the predictions here
        all_predictions = []
#     #print ("Number of batches: %s",len(batches))
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
#
 # Print accuracy if y_test is defined
if y_test is not None:
    print("***************************************")
    print("***********Results*********************")
    #print("y_test: %s",y_test)
    #print("x_test: %s",x_test)
    print("All Predictions:\n")
    print (all_predictions)
    print("--End All Predictions\m")
    print("Y Test")
    print(y_test)
    print("End Y Test")
    print("Length of All Predictions {}".format(len(all_predictions)))
    print("Length of y test {}".format(len(y_test)))
    i = 0
    correct_predictions = 0
    pos_lab = [0, 1]
    neg_lab = [1, 0]
    all_predictions_fixed = []
    for p in all_predictions:
        if ((p == 0) and np.all((y_test[i] == neg_lab))):
            correct_predictions = correct_predictions + 1
            all_predictions_fixed.append(neg_lab)
        if ((p == 1) and np.all((y_test[i] == pos_lab))):
            correct_predictions = correct_predictions + 1
            all_predictions_fixed.append(pos_lab)
        i = i + 1
    print ("Correct Predictions:{}".format(correct_predictions))
    print(all_predictions_fixed)
    all_predictions_fixed = np.array(all_predictions_fixed)
    print(all_predictions_fixed)
    correct_predictions = float(sum(all_predictions_fixed == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("All predictions%S",len(all_predictions_fixed))
    print("y test: %s",len(y_test))
    print("x_test: %s",len(x_test))
    print("Incorrect Predictions %s", len(y_test) - correct_predictions)
    print("Correct Predictions %s", len(y_test) - float(sum(all_predictions_fixed != y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print("Precision, Recall, Fscore")
    print(confusion_matrix(y_test, all_predictions_fixed))
    print(precision_recall_fscore_support(y_test, all_predictions_fixed, average='micro'))
#     outfile.write("\nTotal number of test examples: {}".format(len(y_test)))
#     outfile.write("\nAll predictions {}".format(len(all_predictions)))
#     outfile.write("\ny test: {}".format(len(y_test)))
#     outfile.write("\nx_test: {}".format(len(x_test)))
#     outfile.write("\nIncorrect Predictions {}".format(float(sum(all_predictions != y_test))))
#     outfile.write("\nCorrect Predictions {}".format(len(y_test) - float(sum(all_predictions != y_test))))
#     outfile.write("\nAccuracy: {:g}".format(correct_predictions / float(len(y_test))))
#     outfile.write("\nPrecision, Recall, Fscore")
#     #outfile.write(precision_recall_fscore_support(y_test, all_predictions, average='micro'))
#     outfile.write(np.array2string(confusion_matrix(y_test, all_predictions),separator=','))
#     #outfile.write(confusion_matrix(y_test, all_predictions))
#     outfile.close()
