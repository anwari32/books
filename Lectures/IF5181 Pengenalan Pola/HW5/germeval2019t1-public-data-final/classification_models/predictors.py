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


# Filename: predictors.py
# Authors:  Rami Aly, Steffen Remus and Chris Biemann
# Description: Predictor file used for baseline and contender system as described in the task report for GermEval-2019 Task 1: Shared task on hierarchical classification of blurbs.

from stop_words import get_stop_words
import string
punctuations = string.punctuation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
import re

import string
punctuations = string.punctuation

import spacy
parser = None
stopwords = None


def spacy_init(language):
    """
    Initilizes spacy and stop-words for respective language
    """
    global parser, stopwords
    print("Initialize Spacy")
    if language in ["EN", "WOS"]:
        parser = spacy.load('en_core_web_sm')
        stopwords = get_stop_words('en')
    elif language == 'DE':
        parser = spacy.load('de_core_news_sm')
        stopwords = get_stop_words('de')
    print("Intialization Spacy finished")

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9()!?\'\`äöüß@ ]", "", string)
    return string.strip().lower()

class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}


def identity_tokenizer(text):
    return text


def spacy_tokenizer_basic(sentence):
    """
    Very basic preprocessing(tokenizing, removal of stopwords, too many whitespaces)
    """
    tokens = parser(sentence)
    tokens = [tok.text for tok in tokens]
    tokens = [tok for tok in tokens if tok not in stopwords and " " not in tok]
    return tokens


def vectorizerSpacy():
    cs = TfidfVectorizer(tokenizer = identity_tokenizer, ngram_range=(1,2), lowercase = False)
    return cs
