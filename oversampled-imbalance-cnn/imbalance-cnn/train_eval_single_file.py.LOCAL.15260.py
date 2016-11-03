#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import random
import data_helpers_single_file
import MySQLdb as mdb
import sys
from random import randint
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
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
run_number = 1
t = 6
p = 1
rand_seed = randint(0, 9)
print("Loading data...")
imbalance_size = 5000
pos_or_neg = "positive"
if t == 0:
    imbalance_size = 1500
    pos_or_neg = "positive"
    random_seed = rand_seed
if t == 1:
    imbalance_size = 1500
    pos_or_neg = "negative"
    random_seed = rand_seed
if t == 2:
    imbalance_size = 2500
    pos_or_neg = "positive"
    random_seed = rand_seed
if t == 3:
    imbalance_size = 2500
    pos_or_neg = "negative"
    random_seed = rand_seed
if t == 4:
    imbalance_size = 3500
    pos_or_neg = "positive"
if t == 5:
    imbalance_size = 3500
    pos_or_neg = "negative"
if t == 6:
    imbalance_size = 5000
    pos_or_neg = "positive"

outfile = open('sf_'+str(imbalance_size)+'_' + pos_or_neg + '_results_run_'+str(p)+'.txt', 'w')
dbfieldname = 'sf_'+str(imbalance_size)+'_' + pos_or_neg + '_results_run_'+str(p)
outfile.write("Data Resutls for {} {}".format(imbalance_size,pos_or_neg))
x_text, y = data_helpers_single_file.load_data_and_labels(imbalance_size,pos_or_neg)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(random_seed)
shuffle_indices = np.random.permutation(np.arange(len(y)))
#x_raw_shuffled = x_text[shuffle_indices]
x_shuffled = x[shuffle_indices]
for x in range(0, 2):
    print("The x shuffled is:%s", x_shuffled)
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
#x_raw = x_raw_shuffled[:-1000]

print("Writing out x dev")
np.save('dev_x.txt',x_dev)
print("Writing out y dev")
np.save('dev_y.txt',y_dev)

#Get count of positive negative in train set
pos_value = np.array([0,1])
neg_value = np.array([1,0])

list_positive_instances = []
list_negative_instances = []
list_positive_balanced = []
list_negative_balanced = []
runonceneg = 0
runoncepos = 0
for x in range(0, len(x_train)):
    if (y_train[x]==pos_value).all():
       print("Positive Label")
       list_positive_instances.append(x_train[x])
    else:
       print("Negative label")
       list_negative_instances.append(x_train[x])

#print("This is the new array:%s",x_train[1])
print("The count of positive labels in test: %s",len(list_positive_instances))
print("The count of negative labels in test: %s",len(list_negative_instances))


if len(list_positive_instances) > len(list_negative_instances):
    print("Oversampling the negative instances")
    outfile.write("Oversampling the negative instances")
    for x in range(0,len(list_positive_instances)):
        list_negative_balanced.append(list_negative_instances[random.randint(0,len(list_negative_instances)-1)])
    print("Negative size now: {}".format(len(list_negative_balanced)))
    outfile.write("Negative size now: {}".format(len(list_negative_balanced)))
    list_negative_instances = list_negative_balanced
else:
    print("Oversampling the positive instances")
    outfile.write("Oversampling the positive instances")
    for x in range(0,len(list_negative_instances)):
        list_positive_balanced.append(list_positive_instances[random.randint(0,len(list_positive_instances)-1)])
    print("Positive size now: %s",len(list_positive_balanced))
    outfile.write("Positive size now: {}".format(len(list_positive_balanced)))
    list_positive_instances = list_positive_balanced

#Regenerate the labels

positive_labels = [[0,1] for _ in list_positive_instances]
negative_labels = [[1,0] for _ in list_negative_instances]
print("Length of positive labels:%s",positive_labels)
print("Length of negative labels:%s",negative_labels)

p_length = len(positive_labels)
n_length = len(negative_labels)


y_t = np.concatenate([positive_labels, negative_labels],0)
x_t = np.array(list_positive_instances + list_negative_instances)
#for x in range(0,len(x_t)):
#    print("Instances:%s",x_t[x])

#x_t = [data_helpers.clean_str(sent) for sent in x_t]

# Build vocabulary
#max_document_length = max([len(x.split(" ")) for x in x_t])
#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#x = np.array(list(vocab_processor.fit_transform(x_t)))

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_t)))
x_train = x_t[shuffle_indices]
y_train = y_t[shuffle_indices]
print("Overall Length:%s", len(y_train))
outfile.write("Overall Length:{}".format(len(y_train)))


print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
#print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


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
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
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
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        #if writer:
        #        writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers_single_file.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


print("\n Beginning Dev data load")
#Begin Evaluation of Dev
x_raw = np.load("dev_x.txt.npy")
y_test = np.load("dev_y.txt.npy")
y_test = np.argmax(y_test, axis=1)

# Map data into vocabulary
vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.load("dev_x.txt.npy")

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
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

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers_single_file.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
    #print ("Number of batches: %s",len(batches))
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    print("***************************************")
    print("***********Results*********************")
    #print("y_test: %s",y_test)
    #print("x_test: %s",x_test)
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("All predictions%S",len(all_predictions))
    print("y test: %s",len(y_test))
    print("x_test: %s",len(x_test))
    print("Incorrect Predictions %s", float(sum(all_predictions != y_test)))
    print("Correct Predictions %s", len(y_test) - float(sum(all_predictions != y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print("Precision, Recall, Fscore")
    outfile.write("\nTotal number of test examples: {}".format(len(y_test)))
    outfile.write("\nAll predictions {}".format(len(all_predictions)))
    outfile.write("\ny test: {}".format(len(y_test)))
    outfile.write("\nx_test: {}".format(len(x_test)))
    outfile.write("\nIncorrect Predictions {}".format(float(sum(all_predictions != y_test))))
    outfile.write("\nCorrect Predictions {}".format(len(y_test) - float(sum(all_predictions != y_test))))
    outfile.write("\nAccuracy: {:g}".format(correct_predictions / float(len(y_test))))
    outfile.write("\nPrecision, Recall, Fscore")
    #outfile.write(precision_recall_fscore_support(y_test, all_predictions, average='micro'))
    outfile.write(np.array2string(confusion_matrix(y_test, all_predictions),separator=','))
    #outfile.write(confusion_matrix(y_test, all_predictions))
    outfile.close()
    try:
        con = mdb.connect('localhost', 'datauser', 'datauser', 'tensorflow');

        cur = con.cursor()

        c_matrix = confusion_matrix(y_test, all_predictions)
        data_insert = {
            'name': dbfieldname,
            'imbalance': str(imbalance_size),
            'positive_or_negative': pos_or_neg,
            'train_negative': p_length,
            'train_positive': n_length,
            'true_negative': c_matrix[0][0],
            'false_positive': c_matrix[0][1],
            'false_negative': c_matrix[1][0],
            'true_positive': c_matrix[1][1],
            'accuracy': (correct_predictions / float(len(y_test))),
            'incorrect': (float(sum(all_predictions != y_test))),
            'correct': (len(y_test) - float(sum(all_predictions != y_test))),
            'notes': ''

        }
        sqlInsert = 'Insert into cnn_runs VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        print('\n')
        print(sqlInsert)
        cur.execute(sqlInsert, data_insert)
        #cur.execute("SELECT VERSION()")

        ver = cur.fetchone()

        #print "Database version : %s " % ver

    except mdb.Error, e:

        print "Error %d: %s" % (e.args[0], e.args[1])
        sys.exit(1)

    finally:

        if con:
            con.close()


