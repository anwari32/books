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


# Filename: classification_baseline.py
# Authors:  Rami Aly, Steffen Remus and Chris Biemann
# Description: Baseline system as described in the task report for GermEval-2019 Task 1: Shared task on hierarchical classification of blurbs.


from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
import string
punctuations = string.punctuation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomTreesEmbedding, BaggingClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.externals import joblib
import os
from predictors import predictors, vectorizerSpacy, spacy_init, spacy_tokenizer_basic
from os.path import join
import json
import codecs
import argparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
import io
import pickle
import string
from bs4 import BeautifulSoup
punctuations = string.punctuation
import pickle


CV_NUM = 3


def train_test(classifier, train, train_label, test, train_label_subtask_a, isbns):
    """
    Method for evaluation performance of SVM
    train test pipeline
    """
    data_labels = [train_label_subtask_a, train_label]
    pred_data_all = []
    for i, data_label in enumerate(data_labels):
        vectorizer = vectorizerSpacy()
        lb = MultiLabelBinarizer()
        clas = OneVsRestClassifier(classifier)
        pipe = Pipeline([
                         ('vectorizer', vectorizer),
                         ('classifier',clas)])
        y = lb.fit_transform(data_label)
        pipe.fit(train, y)
        pred_data = pipe.predict(test)

        if i == 1:
            pred_data = adjust_hierarchy(pred_data, lb)
        pred_data = lb.inverse_transform(pred_data)

        pred_data_all.append(pred_data)
    outfile = open('LT-UHH__baseline.txt', 'w')
    for i,predictions in enumerate(pred_data_all):
        if i == 0:
            outfile.write('subtask_a\n')
        elif i == 1:
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


def create_classifier(type):
    if type == "LogisticRegressionL2": clf = LogisticRegression(penalty='l2', tol=0.0001, C=1.0)  #, class_weight='auto')
    elif type == "LogisticRegressionL1": clf = LogisticRegression(penalty='l1', tol=0.0001, C=1.0)  #, class_weight='auto')
    elif type == "MultinomialNB": clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    elif type == 'LinearSVC': clf = LinearSVC(C=1)
    elif type == 'SVC':  clf = SVC(C=50, probability=True,  gamma= 0.0078)
    elif type == 'RandomForest': clf = RandomForestClassifier(n_estimators = 100)
    elif type == 'AdaBoost': clf = AdaBoostClassifier()
    elif type == 'RandomTreesEmbedding': clf = RandomTreesEmbedding()
    elif type == 'Bagging': clf = BaggingClassifier()
    elif type == "SGD": clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter= 5, random_state=42)
    else: clf = LogisticRegression(penalty='l2', tol=0.0001, C=1.0)  # , class_weight='auto')
    return clf


def run(mode, dev):
    spacy_init('DE')
    isbns, train, test = read_input(dev)
    X_train = [entry[0] for entry in train]
    y_train = [entry[1] for entry in train]
    METHODS_MULTI = ["LinearSVC"]
    #["LogisticRegressionL1", "LogisticRegressionL2", "LinearSVC", "Bagging","RandomForest", "MultinomialNB","AdaBoost"]
    for method in METHODS_MULTI:
        sl = create_classifier(method)
        hierarchies,_ = extract_hierarchies()
        y_train_sub_a = [set([label for label in labelset if label in hierarchies[0]]) for labelset in y_train]
        train_test(sl, X_train, y_train, test, y_train_sub_a, isbns)


def read_input(dev):
    if dev:
        data_out_path = 'data_spacy'
    else:
        data_out_path = 'data_spacy_test'
    if not os.path.exists(data_out_path):
        if dev:
            train_file = '../blurbs_dev.txt'
            test_file = '../blurbs_dev_nolabel.txt'
        else:
            train_file = '../blurbs_train_and_dev.txt'
            test_file = '../blurbs_test_nolabel.txt'

        train_data = load_data(train_file, "train")
        test_data = load_data(test_file, "test")
        isbns = load_isbns(test_file)
        train_data = [(spacy_tokenizer_basic(entry[0]), entry[1]) for entry in train_data]
        test_data = [spacy_tokenizer_basic(entry) for entry in test_data]
        data = [isbns, train_data, test_data]
        data_out_file = open(data_out_path, 'wb')
        pickle.dump(data, data_out_file)
    else:
        data_in_file = open(data_out_path, 'rb')
        data = pickle.load(data_in_file)
    return data

def load_isbns(directory):
    isbns = []
    soup = BeautifulSoup(open(join(directory), 'rt').read(), "html.parser")
    for book in soup.findAll('book'):
        book_soup = BeautifulSoup(str(book), "html.parser")
        isbns.append(str(book_soup.find("isbn").string))
    return isbns


def load_data(directory, status):
    """
    Loads labels and blurbs of dataset
    """
    data = []
    soup = BeautifulSoup(open(join(directory), 'rt').read(), "html.parser")
    for book in soup.findAll('book'):
        if status == 'train':
            categories = set([])
            book_soup = BeautifulSoup(str(book), "html.parser")
            for t in book_soup.findAll('topic'):
                categories.add(str(t.string))
            data.append((str(book_soup.find("body").string), categories))
        elif status == 'test':
            book_soup = BeautifulSoup(str(book), "html.parser")
            data.append(str(book_soup.find("body").string))
    return data


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

def adjust_hierarchy(output_b, binarizer):
    """
    Correction of nn predictions by either restrictive or transitive method
    """
    global ml
    print("Adjusting Hierarchy")
    relations,_ = read_relations()
    hierarchy, _ = extract_hierarchies()
    new_output = []
    outputs = binarizer.inverse_transform(output_b)
    for output in outputs:
        labels = set(list(output))
        if len(labels) >= 1:
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
                            labels = labels | set(all_parents)
        new_output.append(tuple(list(labels)))
    return binarizer.transform(new_output)

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

def main():
    global args
    parser = argparse.ArgumentParser(description="MultiLabel classification of blurbs")
    parser.add_argument('--mode', type=str, default='train_test', choices=['train_test','cv'], help="Mode of the system.")
    parser.add_argument('--dev', type=bool, default= False, help="Evaluate on development set")



    args = parser.parse_args()
    print("Mode: ", args.mode)
    run(args.mode, args.dev)



if __name__ == '__main__':
    main()
