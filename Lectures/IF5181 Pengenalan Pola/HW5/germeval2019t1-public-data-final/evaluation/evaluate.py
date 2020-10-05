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


# Filename: evaluate.py
# Authors:  Rami Aly, Steffen Remus and Chris Biemann
# Description: CodaLab's evaluation script for GermEval-2019 Task 1: Shared task on hierarchical classification of blurbs.
#    For more information visit https://competitions.codalab.org/competitions/21226.
# Requires: sklearn. Install it on unix systems by executing: 'pip install -U scikit-learn'
# Script is compatible with Python version 2 and 3
# Execute script by running: 'python evaluate.py [path to input folder] [path to output folder]' with:
#   [path to input folder]: Folder that contains the gold data 'gold.txt' and the submission in the format [Teamname]__[Systemname].txt
#   [path to output folder]: Folder in which the scores will be written into
# Using the given input_folder 'input_dev/' the output created by the script should be:
#   F1-Score Task A: 0.8947368421052632
#   F1-Score Task B: 0.8333333333333334


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import os.path
from utils import subtask_A_evaluation, subtask_B_evaluation
import io

def readfile(fname):
    data_a = {}
    data_b = {}
    current_task = None
    #with open(fname, 'r') as fin:
    try:
        fin = io.open(fname, mode="r", encoding="utf-8")
    except UnicodeError:
        message = "String is not UTF-8"
        print(message, file = sys.stderr)
        raise Exception(message)

    with fin:
        task = fin.readline()
        task = task.strip().lower()

        def read_task(stopping_word):
            data = {}
            for line_i, line in enumerate(fin):
                line = line.strip()
                if not line: # skip empty lines
                    continue
                if line == stopping_word:
                    return data
                content = line.split('\t')

                id = content[0]
                labels = content[1:]
                if not labels:
                    message = "WARNING: ISBN '{:s}' has no label assignment".format(id)
                    print(message, file=sys.stderr)

                if id in data:
                    message = 'Error. Duplicate id {:s}.'.format(id)
                    print(message, file=sys.stderr)
                    raise Exception(message)
                data[id] = labels
            return data

        #Checks the subtask described in header
        if task == 'subtask_a':
            data_a = read_task('subtask_b')
            data_b = read_task('subtask_a')
        elif task == 'subtask_b':
            data_b = read_task('subtask_a')
            data_a = read_task('subtask_b')
        else:
            message = "Error while reading header. Please make sure to specify either subtask_a or subtask_b."
            print(message, file = sys.stderr)
            raise Exception(message)
    return [data_a, data_b]

def allign_sub_to_truth(truth_data, submission_data, task):
    """
    Matches IDs of submission to IDs in the truth file.
    """
    ordered_submission = []
    ordered_truth = []

    for key in submission_data:
        if key not in truth_data:
            message = "Warning. Found unexpected ID: '{:s}' for Task {:s}.".format(key, task)
            print(message, file=sys.stderr)

    for key in truth_data:
        if key not in submission_data:
            message = "Warning. Expected ID '{:s}' for Task {:s}, but not found.".format(key, task)
            if submission_data:
                print(message, file=sys.stderr)
        ordered_truth.append(truth_data[key])
        if key in submission_data:
            ordered_submission.append(submission_data[key])
        else:
            ordered_submission.append([])

    return [ordered_truth, ordered_submission]


if len(sys.argv) < 3:
    print("System Argument(s) missing. Please make sure to specify the input and output directory respectively, e.g:\n \
    python evaluate.py input_dev/ output_dev/", file = sys.stderr)
    sys.exit(0)
input_dir = sys.argv[1]
output_dir = sys.argv[2]

submission_answer_file = ''
for root, dirs, files in os.walk(input_dir):
    for name in files:
        potential_answer_file = name.split("__")
        if len(potential_answer_file) == 2 and name.endswith('.txt'):
            print("Evaluating the following file: ", os.path.join(root, name))
            submission_answer_file = os.path.join(root, name)
            break

if submission_answer_file == '':
    print("Submission file cannot be found. Make sure it is in the right format: [Teamname]__[Systemname].txt \n \
    For more information please visit https://competitions.codalab.org/competitions/21226#learn_the_details-evaluation", file=sys.stderr)
    exit(1)


submit_file = submission_answer_file
truth_file = os.path.join(input_dir, 'gold.txt')

if not os.path.isfile(truth_file):
    print("Gold file doesn't exist!", file=sys.stderr)
    exit(1)

truth_data_a, truth_data_b = readfile(truth_file)

submission_data_a, submission_data_b = readfile(submit_file)

true_output_a, submission_output_a = allign_sub_to_truth(truth_data_a, submission_data_a, 'A')
true_output_b, submission_output_b = allign_sub_to_truth(truth_data_b, submission_data_b, 'B')

recall_a, precision_a, f1_a, acc_a = subtask_A_evaluation(true_output_a, submission_output_a)
set1_score = f1_a
recall_b, precision_b, f1_b, acc_b = subtask_B_evaluation(true_output_b, submission_output_b)
set2_score = f1_b
print("F1-Score Task A:", f1_a, "\nF1-Score Task B:", f1_b)

output_filename = os.path.join(output_dir, 'scores.txt')
output_detailed_filename = os.path.join(output_dir, 'scores.html')

#create output folder if it does not exist already
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#write the scores into scores.txt file
with open(output_filename, 'w') as fout:
    print('correct:1', file=fout)
    print('set1_score:'+ str(set1_score), file=fout)
    print('set2_score:'+ str(set2_score), file=fout)

with open(output_detailed_filename, 'w') as fout:
    print("Subtask_a detailed scores:<br> Recall: '%0.4f' <br> Precision: '%0.4f' <br> F1_micro: '%0.4f'<br> Subset_acc: '%0.4f'<br><br>"%(recall_a, precision_a, f1_a, acc_a), file = fout)
    print("Subtask_b detailed scores:<br> Recall: '%0.4f' <br> Precision: '%0.4f' <br> F1_micro: '%0.4f'<br> Subset_acc: '%0.4f'<br><br>"%(recall_b, precision_b, f1_b, acc_b), file = fout)
print("Scores have been successfully written to:", output_dir)
