This folder contains files and information for participants of the final phase of the GermEval 2019 Task 1 -- Shared task on hierarchical classification of blurbs (Visit: https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc.html).

This folder has the following structure:

data-final
  |
  +--blurbs_train.txt					# Training data in the xml format as described for the task.
  +--blurbs_dev_all.txt         			# Development data in the same format as the training data. (text + categories)
  +--blurbs_dev_nolabel.txt     			# Development data without categories
  +--blurbs_dev_label.txt				# Gold labels for the development data
  +--blurbs_test_nolabel.txt   	 			# Test data. Participants submit predictions for this set.
  +--hierarchy.txt					# File that contains only the hierarchy in form of parent-child relationships
  +--description.pdf					# Description of the dataset
  +--Germeval_Task1_Report.pdf					# Copy of the task description paper from the KONVENS 2019 proceedings: https://corpora.linguistik.uni-erlangen.de/data/konvens/proceedings/
  +--evaluation						# Folder that contains the evaluation script
  |   |
  |   +--input_sample					# Folder with example data. When running the evaluation script the predictions as well as the gold data needs to be put here
  |   |   |
  |   |   +--[Teamname]__[Systemname].txt			# Sample submission data. The prediction created by the participants system
  |   |   +--gold.txt					# Sample gold data (e.g. blurbs_dev_label.txt, or blurbs_test_label.txt)
  |   +--evaluate.py					# The actual evaluation script used to evaluate submissions
  |   +--utils.py
  |
  +--sample_submission					# Sample submission for codalab (the test set with the required naming scheme)
  |   |
  |   +--Funtastic4__SVM_NAIVEBAYES_ensemble1.zip
  |        |
  |        +--Funtastic4__SVM_NAIVEBAYES_ensemble1.txt
  |
  +--classification_models				# Implementation of the baseline and contender system as described in in the task report of this shared task.
  |   |
  |   +--classification_baseline.py
  |   +--classification_contender.py
  |   +--instructions.txt
  |
  +--submissions				# submission files from every participating team with a system description paper submission (papers can be found in the KONVENS 2019 proceedings: https://corpora.linguistik.uni-erlangen.de/data/konvens/proceedings/)
  |   |
  |   +--test-phase-txt
  |   |   |
  |   |   + ...
  |   +--post-test-phase-txt
  |   |   |
  |   |   + ...
  |   +--all_scores_test.tsv
  |   +--all_scores_post-test.tsv
  |
  +--LICENSE						# License file regarding the distribution of the dataset
