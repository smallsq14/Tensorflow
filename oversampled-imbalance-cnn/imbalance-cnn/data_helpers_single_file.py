import numpy as np
import random
import re
import os
import fnmatch
import itertools
from collections import Counter



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(aImbalance,aPolarity):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    #positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    #positive_examples = [s.strip() for s in positive_examples]
    #print("length of positive:%s",len(positive_examples))
    positive_examples = []
    for(dir, dirs, files) in os.walk('./singlefiledatatrimmed/pos'):
            for file in files:
                    if fnmatch.fnmatch(file, '*.txt'):
                        positive_examples.append(open(os.path.join(dir,file),"r").read())
    list_pos_length  = len(positive_examples)
    positive_examples = [s.strip() for s in positive_examples]
    print("length of positive:%s",list_pos_length)
    if aPolarity == "positive":
    	del positive_examples[aImbalance:]
	print("lenght of positive imbalanced:%s",len(positive_examples))
    negative_examples = []
    for (dir, dirs, files) in os.walk('./singlefiledatatrimmed/neg'):
        for file in files:
            if fnmatch.fnmatch(file, '*.txt'):
                negative_examples.append(open(os.path.join(dir, file), "r").read())
    list_pos_length = len(negative_examples)
    negative_examples = [s.strip() for s in negative_examples]
    print("length of negative:%s",len(negative_examples))
    if aPolarity == "negative":
	del negative_examples[aImbalance:]
	print("lenght of negative imbalanced:%s",len(negative_examples))
    x_text = positive_examples + negative_examples
    #for x in range(0,len(x_text)):
    #    print("Original x-text:%s",x_text)
    x_text = [clean_str(sent) for sent in x_text]
    
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


load_data_and_labels(1500,"positive")
