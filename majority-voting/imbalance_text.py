
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

all_model_predictions = list()
# Data Preparation
# ==================================================

# Load data
random_seed = 10
number_of_classifiers = 10
run_accuracy = []


x_text, y = data_helpers.load_data_and_labels()

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


classifier_list = []

pos_value = np.array([0, 1])
neg_value = np.array([1, 0])

list_positive_instances = []
list_negative_instances = []
imbalance_size = 500
#print("Size of Positive Instances{}".format(len(list_positive_instances)))
positive_test_size = round(.20 * imbalance_size)
positive_train_size = round(imbalance_size - positive_test_size)
negative_test_size = round(.20 * 5331)
negative_train_size = round(5331 - negative_test_size)

print("Positive Test Size{}".format(positive_test_size))
print("Negative Test Size{}".format(negative_test_size))
print("Positive Train Size{}".format(positive_train_size))
print("Negative Train Size{}".format(negative_train_size))

list_pos_train_instances = []
list_neg_train_instances = []
list_pos_dev_instances = []
list_neg_dev_instances = []
pos_counter = 0
pos_second_counter = 0
neg_counter = 0
for x in range(0, len(x_shuffled)):
    if (y_shuffled[x] == pos_value).all():
        if(pos_counter<positive_test_size):
            list_pos_dev_instances.append(x_shuffled[x])
            pos_counter = pos_counter + 1
        else:
            if(pos_second_counter<positive_train_size):
                list_pos_train_instances.append(x_shuffled[x])
                pos_second_counter = pos_second_counter + 1
    else:
        if(neg_counter<negative_test_size):
            list_neg_dev_instances.append(x_shuffled[x])
            neg_counter = neg_counter + 1
        else:
            list_neg_train_instances.append(x_shuffled[x])
#final value
print("New Positive Test Size{}".format(len(list_pos_dev_instances)))
print("New Negative Test Size{}".format(len(list_neg_dev_instances)))
print("New Positive Train Size{}".format(len(list_pos_train_instances)))
print("New Negative Train Size{}".format(len(list_neg_train_instances)))



print("Length of Cut positive dev:%s", len(list_pos_dev_instances))
print("Length of Cut negative dev:%s", len(list_neg_dev_instances))
p_labels = [[0, 1] for _ in list_pos_dev_instances]
n_labels = [[1, 0] for _ in list_neg_dev_instances]
y_dev = np.concatenate([p_labels, n_labels], 0)
x_dev = np.array(list_pos_dev_instances + list_neg_dev_instances)
print("Length of Cut positive test:%s", len(list_pos_dev_instances))
print("Length of Cut negative test:%s", len(list_neg_dev_instances))
list_positive_instances = list_pos_train_instances
list_negative_instances = list_neg_train_instances

print("Length of Cut positive train:%s", len(list_positive_instances))
print("Length of Cut negative train:%s", len(list_negative_instances))

print("Length of Dev X :%s", len(y_dev))
print("Length of Dev Y :%s", len(x_dev))



run_accuracy = []
x_train = []
y_train = []
all_predictions = []
positive_labels = []
negative_labels = []

checkpoint_dir_for_eval = ""

            #no change
y_test = y_dev
x_test = x_dev
rand_seed = int(5)
print("running Method A")
print("positive labels should be empty:{}".format(len(positive_labels)))
print("negative labels should be empty:{}".format(len(negative_labels)))
pos_labels = [[0, 1] for _ in list_positive_instances]
neg_labels = [[1, 0] for _ in list_negative_instances]
print("Length of positive labels:%s", len(pos_labels))
print("Length of negative labels:%s", len(neg_labels))
y_t0 = np.concatenate([pos_labels, neg_labels], 0)
x_t0 = np.array(list_positive_instances + list_negative_instances)
np.random.seed(rand_seed)
shuffle_indices = np.random.permutation(np.arange(len(y_t)))
x_train = x_t0[shuffle_indices]
y_train = y_t0[shuffle_indices]
print("Checking length of x_train {}".format(len(x_train)))
print("Checking length of y_dev {}".format(len(y_dev)))
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
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name),
                                                     tf.nn.zero_fraction(g))
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
                dev_step(x_test, y_test, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))