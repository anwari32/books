#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/python
# coding: utf8

# Copyright 2019 Language Technology Group, Universität Hamburg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Filename: classification_contender.py
# Authors:  Rami Aly, Steffen Remus and Chris Biemann
# Description: Contender system as described in the task report for GermEval-2019 Task 1: Shared task on hierarchical classification of blurbs.
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"  # specify which GPU(s) to be used
from keras.constraints import max_norm
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.models import Model
import keras.losses
from keras.layers import Embedding
from keras import layers, models
from capsulelayers import CapsuleLayer, PrimaryCap, Length
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from sklearn.preprocessing import MultiLabelBinarizer
from keras.callbacks import Callback, EarlyStopping
from predictors import predictors, vectorizerSpacy, spacy_init, spacy_tokenizer_basic, clean_str
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import gensim
import sys
import pickle
import argparse
import codecs
import json
from bs4 import BeautifulSoup
import string
punctuations = string.punctuation
import pickle
from os.path import join
import itertools
import math
import operator
import numpy as np

from collections import Counter
from keras.callbacks import LearningRateScheduler

args = None
ml = MultiLabelBinarizer()

UNSEEN_STRING = "-EMPTY-"

def create_model_capsule(embedding_dim, sequence_length, num_classes, use_static, init_layer, vocabulary, learning_rate,  dense_capsule_dim, n_channels, routings, dev):
    """
    Implementation of capsule network
    """
    over_time_conv = 80
    inputs = Input(shape=(sequence_length,), dtype='int32')

    embedding = pre_embedding(embedding_dim = embedding_dim, seq_length = sequence_length,
        input = inputs, use_static = use_static, voc = vocabulary, dev = dev)

    primarycaps = PrimaryCap(embedding, dim_capsule=8, n_channels= n_channels, kernel_size=over_time_conv,
        strides=1, padding='valid', name = 'primarycaps')

    dense = CapsuleLayer(num_capsule=num_classes, dim_capsule=dense_capsule_dim, routings=routings,
                             name='digitcaps')(primarycaps)

    out_caps = Length(name='capsnet')(dense)
    model = Model(inputs=inputs, outputs=out_caps)

    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=[margin_loss],
                  metrics=['categorical_accuracy'])

    #initilizes the weight of transformation matrix W
    if init_layer:
        weights = model.layers[-2].get_weights()[0]
        co_occurences = co_occurence_weights(weights[0].shape[1], num_classes)
        print(len(co_occurences), len(co_occurences[0]))
        for i, co_occurence in enumerate(co_occurences):
            if i >= weights.shape[1]:
                break
            for j, weight in enumerate(co_occurence):
                #initilzes the  weights between dim of primary and one complete  dense capsule
                weights[j][i][0] = weights[j][i][0] if weight != 0 else 0
        model.layers[-2].set_weights([weights])
    print(model.summary())
    return model

def co_occurence_weights(num_units, num_classes):
    """
     loads the co-occurence matrix with respective weights
    """
    parent_child = []
    w = math.sqrt(6) / math.sqrt(num_units + num_classes)
    occurences = read_all_genres()
    for occurence in occurences:
        #occurence = ml.inverse_transform(occurence)
        if occurence[0].issubset(set(ml.classes_)):
            frequency = occurence[1]
            w_f = w # * math.sqrt(frequency)
            binary_rel = ml.transform([occurence[0]])
            parent_child.append([w_f if x==1 else 0 for x in binary_rel[0]])
    print(len(occurences))
    return parent_child

def read_all_genres():
    occurences = []
    frequency = []
    co_occurences_path =  os.path.join(os.path.dirname(__file__), 'co_occurences')
    if os.path.exists(co_occurences_path):
        co_occurences_file = open(co_occurences_path, 'rb')
        occurences, frequency = pickle.load(co_occurences_file)
    else:
        train = data['y_train_raw']
        for genres in train:
            if genres in occurences:
                frequency[occurences.index(genres)] +=1
            else:
                occurences.append(genres)
                frequency.append(1)
        co_occurence_file = open(co_occurences_path, 'wb')
        pickle.dump([occurences,frequency], co_occurence_file)

    occurences = zip(occurences, frequency)
    occurences = sorted(occurences, key=operator.itemgetter(1), reverse = True)

    return occurences


def margin_loss(y_true, y_pred):
    """
    Margin loss as described in Sabour et al. (2017)
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def pre_embedding(embedding_dim, seq_length, use_static, voc, input, dev, model = None):
    """
    Loads mebedding for model
    """
    if dev:
        embed_saved_path =  os.path.join(os.path.dirname(__file__), 'embed_' + str(seq_length) + '_' + "validation_meta")
    else:
        embed_saved_path =  os.path.join(os.path.dirname(__file__), 'embed_' + str(seq_length) + '_' + "test_meta")
    if os.path.exists(embed_saved_path):
        print("Loading Embedding Matrix...")
        embed_saved_file = open(embed_saved_path, 'rb')
        embedding_matrix = pickle.load(embed_saved_file)
    else:
        w2v_dir = os.path.join(os.path.dirname(__file__), 'wiki.de.vec')
        print("Load Embedding File: ", w2v_dir)
        w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_dir, binary=False)

        print("Embedding Voc Size", len(w2v.wv.vocab))
        print("The unseen string -EMPTY- is not in the embedding: ", "-EMPTY-" not in w2v.wv.vocab)
        count = 0
        embedding_matrix = np.random.uniform(-0.25, 0.25, (len(voc) + 1, embedding_dim))

        for word, i in voc.items():

            if word not in  w2v.wv.vocab:
                continue
            embedding_vector = w2v.wv.get_vector(word)

            if embedding_vector is not None:
                count+=1
                embedding_matrix[i] = embedding_vector

        print("Found: ", count, " words")
        print("Out of", len(voc.items()), "Words in dataset")
        embed_saved_file = open(embed_saved_path, 'wb')
        pickle.dump(embedding_matrix, embed_saved_file)

    trainable = not use_static
    if input != None:
        embedding =  Embedding(len(voc) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=seq_length,
                                trainable= trainable)(input)
        return embedding
    elif model != None:
        model.add(Embedding(len(voc) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=seq_length,
                                trainable= trainable))

class Metrics_eval(Callback):
    """
    Callback to receive score after each epoch of training
    """
    def __init__(self,validation_data):
        self.val_data = validation_data

    def eval_metrics(self):
        #dont use term validation_data, name is reserved
        val_data = self.val_data
        X_test = val_data[0]
        y_test = val_data[1]
        output = self.model.predict(X_test, batch_size = args.batch_size)
        for pred_i in output:
            pred_i[pred_i >=args.activation_th] = 1
            pred_i[pred_i < args.activation_th] = 0
        output = adjust_hierarchy(output)

        return [f1_score(y_test, output, average='micro'),f1_score(y_test, output, average='macro'),
         recall_score(y_test, output, average='micro'),precision_score(y_test, output, average='micro'),
          accuracy_score(y_test, output)]

    def on_epoch_end(self, epoch, logs={}):
        f1, f1_macro, recall, precision, acc = self.eval_metrics()
        print("For epoch %d the scores are F1: %0.4f, Recall: %0.2f, Precision: %0.2f, acc: %0.4f, F1_m: %0.4f"%(epoch, f1, recall, precision, acc, f1_macro))
        print((str(precision) + '\n' +  str(recall) + '\n' +
                 str(f1_macro) + '\n' + str(f1) + '\n' + str(acc)).replace(".", ","))


def train(model, early_stopping = True, validation = True):
    """
    Trains a neural network, can use early_stopping and validationsets
    """

    print("Traning Model...")
    callbacks_list = []
    lr_decay = LearningRateScheduler(schedule=lambda epoch: args.learning_rate * (args.learning_decay ** epoch))
    callbacks_list.append(lr_decay)
    if early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=0, mode='auto')
        callbacks_list.append(early_stopping)
    if validation:
        metrics_callback = Metrics_eval(validation_data = (data['X_test'], data['y_test']))
        callbacks_list.append(metrics_callback)
        model.fit(data['X_train'], data['y_train'], batch_size=args.batch_size, epochs=args.epochs,
         verbose=1, callbacks=callbacks_list, validation_data=(data['X_test'], data['y_test'])) # starts training
    else:
        metrics_callback = Metrics_eval(validation_data = (data['X_train'], data['y_train']))
        callbacks_list.append(metrics_callback)
        model.fit(data['X_train'], data['y_train'], batch_size=args.batch_size, epochs=args.epochs, verbose=1,
         callbacks = callbacks_list)

def test(model, data_l, label_all):
    """
    Tests a neural network on the given data
    """
    global data
    print("Testing Model...")
    print(len(data_l))

    output = model.predict(data_l, batch_size = args.batch_size)
    binary_output = np.array(output, copy = True)
    #print(binary_output)
    for pred_i in output:
        pred_i[pred_i >=args.activation_th] = 1
        pred_i[pred_i < args.activation_th] = 0

    output = adjust_hierarchy(output)
    predictions =  ml.inverse_transform(output)

    isbns = data['isbns']
    outfile = open('LT-UHH__contender.txt', 'w')
    outfile.write('subtask_a\n')
    relations =  read_relations()[0]

    for j, entry in enumerate(predictions):
        if len(entry) == 0:
            continue
        outfile.write(str(isbns[j]) + '\t')
        for i, label in enumerate(entry):
            if get_level_genre(relations, label)[0] != 0:
                if i == len(entry) -1:
                    outfile.write('\n')
                continue
            if i == len(entry) -1:
                outfile.write(str(label) + '\n')
            else:
                outfile.write(str(label) + '\t')
    outfile.write('subtask_b\n')
    for j, entry in enumerate(predictions):
        if len(entry) == 0:
            continue
        outfile.write(str(isbns[j]) + '\t')
        for i, label in enumerate(entry):
            if i == len(entry) -1:
                outfile.write(str(label) + '\n')
            else:
                outfile.write(str(label) + '\t')

    f1 = f1_score(label_all, output, average='micro')
    f1_macro = f1_score(label_all, output, average='macro')
    recall = recall_score(label_all, output, average='micro')
    precision =  precision_score(label_all, output, average='micro')
    accuracy = accuracy_score(label_all, output)
    print("F1: " + str(f1))
    print("F1_macro: " + str(f1_macro))
    print("Recall: " + str(recall))
    print("Precision: " + str(precision))
    print("Accuracy: " + str(accuracy))


def adjust_hierarchy(output_b):
    global ml
    print("Adjusting Hierarchy")
    relations,_ = read_relations()
    hierarchy, _ = extract_hierarchies()
    new_output = []
    outputs = ml.inverse_transform(output_b)
    for output in outputs:
        labels = set(list(output))
        #print(labels)
        if len(labels) > 1:
            labels_cp = labels.copy()
            labels_hierarchy = {}
            for level in hierarchy:
                for label in labels:
                    if label in hierarchy[level]:
                        if level in labels_hierarchy:
                            labels_hierarchy[level].add(label)
                        else:
                            labels_hierarchy[level] = set([label])
            for level in labels_hierarchy:
                if level > 0:
                    for label in labels_hierarchy[level]:
                        all_parents = get_parents(label, relations)
                        missing = [parent for parent in all_parents if not parent in labels]
                        no_root = True
                        for element in missing:
                            if element in labels and get_genre_level(element, hierarchy) == 0:
                                labels = labels | all_parents
                                no_root = False

                        if len(missing) >= 1:
                            old_labels = labels.copy()
                            labels = labels | set(all_parents)
        new_output.append(tuple(list(labels)))
    return ml.transform(new_output)

def get_parents(child, relations):
    """
    Get the parent of a genre
    """
    parents = []
    current_parent = child
    last_parent = None
    while current_parent != last_parent:
        last_parent = current_parent
        for relation in relations:
            if relation[1] == current_parent:
                current_parent = relation[0]
                parents.append(current_parent)
    return parents

def extract_hierarchies():
    """
    Returns dictionary with level and respective genres
    """
    hierarchies_inv = {}
    relations, singletons = read_relations()
    genres = set([relation[0] for relation in relations] +
    [relation[1] for relation in relations]) | singletons
    #genres, _= read_all_genres(language, max_h)
    for genre in genres:
        if not genre in hierarchies_inv:
            hierarchies_inv[genre] = 0
    for genre in genres:
        hierarchies_inv[genre], _ = get_level_genre(relations, genre)
    hierarchies = {}
    for key,value in hierarchies_inv.items():
        if not value in hierarchies:
            hierarchies[value] = [key]
        else:
            hierarchies[value].append(key)
    return [hierarchies,hierarchies_inv]

def read_relations():
    """
    Loads hierarchy file and returns set of relations
    """
    relations = set([])
    singeltons = set([])
    REL_FILE =  '../hierarchy.txt'
    with open(REL_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            rel = line.split('\t')
            if len(rel) > 1:
                rel = (rel[0], rel[1][:-1])
            else:
                singeltons.add(rel[0][:-1])
                continue
            relations.add(rel)
    return [relations, singeltons]

def get_level_genre(relations, genre):
    """
    return hierarchy level of genre
    """
    height = 0
    curr_genre = genre
    last_genre = None
    while curr_genre != last_genre:
        last_genre = curr_genre
        for relation in relations:
            if relation[1] == curr_genre:
                height+=1
                curr_genre = relation[0]
                break
    return height, curr_genre

def remove_low_freq_words(blurb, freq_dict):
    MIN_FRE = 2
    sentence = blurb.copy()
    for i,word in enumerate(blurb):
        if freq_dict[word] < MIN_FRE:
            sentence[i] = UNSEEN_STRING
    return sentence

def read_input(dev):
    """
    Loads and preprocesses dataset. Loads either the development set as the test set or the test set for the final evaluation.
    """
    data = {}
    if dev:
        data_out_path = 'data_spacy_capsule_meta'
        train_file = '../blurbs_train.txt'
        test_file = '../blurbs_dev.txt'
    else:
        data_out_path = 'data_spacy_test_capsule_meta'
        train_file = '../blurbs_train_and_dev.txt'
        test_file = '../blurbs_test.txt'

    if not os.path.exists(data_out_path):

        train_data = load_data(train_file)
        test_data = load_data(test_file)

        train_data = [(spacy_tokenizer_basic(clean_str(entry[0])), entry[1]) for entry in train_data]
        test_data = [(spacy_tokenizer_basic(clean_str(entry[0])), entry[1]) for entry in test_data]

        freq = {}
        for blurb in [entry[0] for entry in train_data] + [entry[0] for entry in test_data]:
            for word in blurb:
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word]+=1
        train_data = [(remove_low_freq_words(entry[0], freq), entry[1]) for entry in train_data]
        test_data = [(remove_low_freq_words(entry[0], freq), entry[1]) for entry in test_data]


        data['X_train'] = [entry[0] for entry in train_data]
        data['y_train'] = [entry[1] for entry in train_data]
        data['X_test'] = [entry[0] for entry in test_data]
        data['y_test'] = [entry[1] for entry in test_data]



        sentences_padded_train = pad_sequences(data['X_train'], maxlen=args.sequence_length, dtype='str', padding = 'post', truncating ='post')
        vocabulary_train, vocabulary_inv_train = build_vocab(sentences_padded_train)
        data['X_train'] = build_input_data(sentences_padded_train, vocabulary_train)

        data['X_test'] = [[a if a in vocabulary_train else UNSEEN_STRING for a in sentence] for sentence in data['X_test']]
        sentences_padded_test = pad_sequences(data['X_test'], maxlen=args.sequence_length, dtype='str', padding = 'post', truncating ='post')
        data['X_test'] = build_input_data(sentences_padded_test, vocabulary_train)
        data['vocabulary'] = vocabulary_train
        data['vocabulary_inv'] = vocabulary_inv_train

        data_out_file = open(data_out_path, 'wb')
        pickle.dump(data, data_out_file)
        global ml
        data['y_train_raw'] = data['y_train'].copy()
        data['y_train'] = ml.fit_transform(data['y_train'])
        y_test = [[x for x in sample if x in ml.classes_] for sample in data['y_test']]
        data['y_test'] = ml.transform(y_test)

    else:
        data_in_file = open(data_out_path, 'rb')
        data = pickle.load(data_in_file)
        global ml
        data['y_train_raw'] = data['y_train'].copy()
        data['y_train'] = ml.fit_transform(data['y_train'])
        y_test = [[x for x in sample if x in ml.classes_] for sample in data['y_test']]
        data['y_test'] = ml.transform(y_test)
        data['isbns'] = load_isbns(test_file)

    print("First entries in the dataset...", data['X_train'][:5])
    return data

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index    print ml.classes_
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    vocabulary[UNSEEN_STRING] = len(vocabulary_inv)
    #print vocabulary_inv
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x


def load_isbns(directory):
    isbns = []
    soup = BeautifulSoup(open(join(directory), 'rt').read(), "html.parser")
    for book in soup.findAll('book'):
        book_soup = BeautifulSoup(str(book), "html.parser")
        isbns.append(str(book_soup.find("isbn").string))
    return isbns


def load_data(directory):
    """
    Loads labels and blurbs of dataset
    """
    data = []
    soup = BeautifulSoup(open(join(directory), 'rt').read(), "html.parser")
    for i,book in enumerate(soup.findAll('book')):
        # if i > 100:
        #     break
        categories = set([])
        book_soup = BeautifulSoup(str(book), "html.parser")
        for t in book_soup.findAll('topic'):
            categories.add(str(t.string))
        blurb = (str(book_soup.find("body").string))
        title = (str(book_soup.find("title").string))
        author = (str(book_soup.find("authors").string))
        text = title + ' @ ' + author + ' @ ' + blurb
        data.append((text, categories))
    return data

def run():
    spacy_init('DE')
    loaded_data = read_input(args.dev)
    global data
    data = loaded_data

    capsule_a = create_model_capsule(args.embed_dim, args.sequence_length, len(data['y_train'][0]),
      args.use_static, args.init_layer, data['vocabulary'], args.learning_rate,
      args.dense_capsule_dim, args.capsule_num, 3, args.dev)
    train(capsule_a,  early_stopping = args.use_early_stop, validation = False)
    test(capsule_a, data_l = data['X_test'], label = data['y_test'])


def main():
    global args
    parser = argparse.ArgumentParser(description="MultiLabel classification of blurbs")
    parser.add_argument('--mode', type=str, default='train_test', choices=['train_test','cv'], help="Mode of the system.")
    parser.add_argument('--dev', action='store_true', default = False , help = 'Use dev set')
    parser.add_argument('--dense_capsule_dim', type=int, default=8, help = 'Capsule dim of dense layer')
    parser.add_argument('--capsule_num', type=int, default=100, help = 'number channels of primary capsules')
    parser.add_argument('--batch_size', type=int, default=32, help = 'Set minibatch size')
    parser.add_argument('--use_static', action='store_true', default=False, help = "Use static embeddings")
    parser.add_argument('--sequence_length', type=int, default=100, help = "Maximum sequence length")
    parser.add_argument('--epochs', type=int, default=10, help = "Number of epochs to run")
    parser.add_argument('--init_layer', action='store_true', default=False, help = "Init final layer with cooccurence")
    parser.add_argument('--embed_dim', type=int, default=300, help = "Embedding dim size")
    parser.add_argument('--use_early_stop', action='store_true', default = False , help = 'Activate early stopping')
    parser.add_argument('--learning_decay', type=float, default = 1., help = 'Use decay in learning, 1 is None')
    parser.add_argument('--learning_rate', type = float, default = 0.002, help = 'Set learning rate of network')
    parser.add_argument('--activation_th', type=float, default=0.5, help = "Activation Threshold of output")


    args_l = parser.parse_args()
    global args
    args = args_l
    print("Mode: ", args.mode)
    run()

if __name__ == '__main__':
    main()
