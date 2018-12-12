#!/usr/bin/env python3
import sys
import string
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

giveaways = {}

# parse giveaway words
# ARGUMENTS:
#    - giveaway_filename: file containing giveaway words
# RETURNS:
#    - set formed from words in the file
def parsegiveaway(giveaway_filename):
	giveaways = set()
	file = open(giveaway_filename, 'r')
	for line in file:
		word = line[:-1] #strip '\n'
		giveaways.add(word)

	return giveaways

# get features of voicemail transcription
# ARGUMENTS:
#    - line: line of transcription text
#    - giveaways: array-like container of giveaway words
# RETURNS:
#    - parsed feature vector features
def parse(line, giveaways):
	pauses = set(['um','uh','erm','eh'])
	cleanline = ''
	punctuationCount = 0
	for c in line:
		if c in string.punctuation:
			punctuationCount += 1
		else:
			cleanline += c

	cleanline = cleanline.split()
	umCount , signalwordCount, wordCount = 0 , 0 , len(cleanline)
	for word in cleanline:
		if word in pauses:
			umCount += 1
		elif word in giveaways:
			signalwordCount += 1

	return wordCount , signalwordCount , umCount , punctuationCount

# form feature vectors
# ARGUMENTS:
#    - giveaways: array-like container of giveaway words
#    - label: boolean indicating human(0) or robo(1)
#    - filename: filename containing data examples
#    - lines: string containing data examples
#    Either specifiy a filename or provide a line, but not both.
# RETURNS:
#    list of feature vectors
def extracttrain(giveaways, label, filename=None, lines=None):
	# you're supposed to either put the data in a file or data structure
	if filename is not None and lines is not None:
		print('Error: either put data in file or data structure, not both')
		exit()

	features = []
	if filename is None:
		for line in lines:
			label = int(line[0])
			sentence = ' '.join(line[1:].strip().split())
			if sentence == '':
				continue
			sentence = sentence.lower()
			wordCount, signalwordCount, umCount, punctuationCount = parse(sentence, giveaways)
			features.append([wordCount, signalwordCount, umCount, punctuationCount , label ])

	else:
		file = open(filename, 'r')

		for line in file:
			line = (line[:-1]).lower()
			wordCount , signalwordCount , umCount , punctuationCount = parse(line, giveaways)
			features.append([wordCount, signalwordCount, umCount, punctuationCount , label ])
	return features

# extract features from test data
# ARGUMENTS:
#    - testfilename: file containing test data
# RETURNS:
#    - features: list of feature vectors
#    - labels: list of labels
def extracttest(testfilename):
	testfile = open(testfilename , 'r')
	features , labels = [] , []

	for line in testfile:
		line = line[:-1]
		if line.split()[0] == 'ROBO':
			labels.append(1)
		else:
			labels.append(0)
		wordCount , signalwordCount , umCount , punctuationCount = parse(line[8:].lower())

		features.append([wordCount,signalwordCount,umCount,punctuationCount])

	return features , labels

# classify using individually tuned, bootstrapped trees
# ARGUMENTS:
#    - traindata: list of feature vectors of the training set
#    - trainlabel: labels corresponding to traindata
#    - testfeatures: list of feature vectors of the test set
#    - num_bootstrap: how many bootstrapped trees to execute
# RETURNS:
#    - dictionary mapping {test feature vector: prediction}
def bootstrap_tree(traindata, trainlabel, testfeatures, num_bootstrap):
	pred_dict = {}
	for i in range(0, num_bootstrap):
		# sample with replacement from the training data
		traindata, trainlabel = zip(*random.choices(list(zip(traindata, trainlabel)), k=len(traindata)))
		traindata = list(traindata)
		trainlabel = list(trainlabel)
		# tune a decision tree classifier
		params = {'max_depth': np.arange(1,33, 5),
              	'min_samples_split': np.arange(2,11,2),
              	'min_samples_leaf': np.arange(1,8,2),
              	'min_weight_fraction_leaf': np.arange(0.0,0.6,0.1)
    	}
		gs = GridSearchCV(estimator=tree.DecisionTreeClassifier(),
                      	param_grid=params,
                      	scoring='accuracy',
                      	n_jobs=4,
                      	verbose=0)
		gs.fit(traindata, trainlabel)
		clf = tree.DecisionTreeClassifier()
		clf.set_params(**gs.best_params_)
		clf = clf.fit(traindata, trainlabel)
		predictions = clf.predict(testfeatures)
		for t, p in zip(testfeatures, predictions):
			if tuple(t) not in pred_dict:
				pred_dict[tuple(t)] = p
			else:
				pred_dict[tuple(t)] += p
	return pred_dict

# randomly shuffle the data and separate the feature vectors from their labels
# ARGUMENTS:
#    - data: list of feature vectors with their labels appended to the end
# RETURNS:
#    - traindata: list of feature vectors of the training data
#    - trainlabel: list of labels corresponding to traindata
#    - testfeatures: list of feature vectors of the test data
#    - testlabels: list of labelse corresponding to testfeatures
def get_random_data(data):
	np.random.shuffle(data)

	traindata , trainlabel = [] , []

	#testfeatures , testlabels = extracttest(testfilename)
	testfeatures = []
	testlabels = []
	i = 0
	for fv in data:
		if i < len(data)*0.8:
			traindata.append(fv[:-1])
			trainlabel.append(fv[-1])
		else:
			testfeatures.append(fv[:-1])
			testlabels.append(fv[-1])
		i += 1

	return traindata, trainlabel, testfeatures, testlabels

# predict spam/ham test messages and print metrics
# ARGUMENTS:
#    - train_file: file name of training data
#    - test_file: file name of test data
#    - giveaways: array-like container of giveaway words
def spam_ham_predict(train_file, test_file, giveaways):
	# read in the messages and parse
	train_lines = []
	with open(train_file, 'r', encoding='ISO-8859-1') as tr:
		for line in tr.readlines():
			train_lines.append(line.strip())
	train_data = extracttrain(giveaways, 0, lines=train_lines)
	np.random.shuffle(train_data)

	# split training data into features and labels
	train_features, train_labels = [], []
	for fv in train_data:
		train_features.append(fv[:-1])
		train_labels.append(fv[-1])

	# prepare test data
	test_lines= []
	with open(test_file, 'r', encoding='ISO-8859-1') as tf:
		for line in tf.readlines():
			test_lines.append(line.strip())
	test_data = extracttrain(giveaways, 1, lines=test_lines)
	test_features, test_labels = [], []
	for fv in test_data:
		test_features.append(fv[:-1])
		test_labels.append(fv[-1])

	# fit the tree to the data we just generated
	# uncomment to tune the tree
	'''
	params = {'max_depth': np.arange(1,33, 5),
              'min_samples_split': np.arange(2,11,2),
              'min_samples_leaf': np.arange(1,8,2),
              'min_weight_fraction_leaf': np.arange(0.0,0.6,0.1)
    }
	gs = GridSearchCV(estimator=tree.DecisionTreeClassifier(),
                      param_grid=params,
                      scoring='accuracy',
                      n_jobs=4,
                      verbose=0)
	gs.fit(traindata, trainlabel)
	'''
	clt = tree.DecisionTreeClassifier()
	clt.fit(train_features, train_labels)
	preds = clt.predict(test_features)
	num_correct = 0
	for pred, actual in zip(preds, test_labels):
		if pred == actual:
			num_correct += 1
	accuracy = num_correct / len(test_labels)
	print('spam/ham accuracy:', accuracy)
	precision = precision_score(test_labels, preds)
	recall = recall_score(test_labels, preds)
	print('precision:', precision)
	print('recall:', recall)

	return


# Run model where the majority vote is taken from
# all the different types of predictors
# ARGUMENTS:
#    - x_train: list of feature vectors of the training data
#    - y_train: list of labels corresponding to x_train
#    - x_test: list of feature vectors of the test data
#    - y_test: list of labels corresponding to x_test
# RETURNS:
#    - accuracy, precision, and recall of the classifier

def majority_vote(x_train, y_train, x_test, y_test):
	# try a bunch of different stuff from sklearn
	logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(x_train, y_train)
	logreg_preds = list(logreg.predict(x_test))

	# adaboost model where the individual units are instantiations of the logreg estimator above
	adaboost = AdaBoostClassifier(base_estimator=logreg, learning_rate=0.3).fit(x_train, y_train)
	ada_preds = list(adaboost.predict(x_test))

	# random forest
	rf = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_split=4, min_samples_leaf=2, max_features=None)
	rf = rf.fit(x_train, y_train)
	rf_preds = list(rf.predict(x_test))

	# decision tree
	dt = tree.DecisionTreeClassifier(max_depth=6, min_samples_split=4, min_samples_leaf=2, max_features=None)
	dt = dt.fit(x_train, y_train)
	dt_preds = list(dt.predict(x_test))

	assert(len(logreg_preds) == len(ada_preds) == len(rf_preds) == len(dt_preds))
	aggregate_pred = [sum(x) for x in zip(logreg_preds, ada_preds, rf_preds, dt_preds)]

	final_pred = [0]*len(aggregate_pred)
	for i in range(len(final_pred)):
		if aggregate_pred[i] > 2:
			final_pred[i] = 1

	accuracy = accuracy_score(y_test, final_pred)
	prec = precision_score(y_test, final_pred)
	recall = recall_score(y_test, final_pred)

	return accuracy, prec, recall

# Run ensemble model
# DESIGN: 
#    - first level: logreg, adaboost, random forest
#    - second level: decision tree
# ARGUMENTS:
#    - x_train: list of feature vectors of the training data
#    - y_train: list of labels corresponding to x_train
#    - x_test: list of feature vectors of the test data
#    - y_test: list of labels corresponding to x_test
# RETURNS:
#    - accuracy, precision, and recall of the classifier

def ensemble(x_train, y_train, x_test, y_test):
	# split the training data into its own training and test data
	prelim_train_data, prelim_test_data = train_test_split(list(zip(x_train, y_train)))
	prelim_train, prelim_train_label, prelim_test, prelim_test_label = [], [], [], []

	for i in range(len(prelim_train_data)):
		prelim_train.append(prelim_train_data[i][0])
		prelim_train_label.append(prelim_train_data[i][1])
	assert(len(prelim_train) == len(prelim_train_label))

	for i in range(len(prelim_test_data)):
		prelim_test.append(prelim_test_data[i][0])
		prelim_test_label.append(prelim_test_data[i][1])
	assert(len(prelim_test) == len(prelim_test_label))

	# try a bunch of different stuff from sklearn
	logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(prelim_train, prelim_train_label)
	logreg_prelim = list(logreg.predict(prelim_test))
	logreg_y = list(logreg.predict(x_test))

	# adaboost model where the individual units are instantiations of the logreg estimator above
	adaboost = AdaBoostClassifier(base_estimator=logreg, learning_rate=0.3).fit(prelim_train, prelim_train_label)
	ada_prelim = list(adaboost.predict(prelim_test))
	ada_y = list(logreg.predict(x_test))

	# random forest
	rf = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_split=4, min_samples_leaf=2, max_features=None)
	rf = rf.fit(prelim_train, prelim_train_label)
	rf_prelim = list(rf.predict(prelim_test))
	rf_y = list(logreg.predict(x_test))

	# use decision tree as the 2nd layer
	info_for_dt = []
	for i in range(len(logreg_prelim)):
		info_for_dt.append([logreg_prelim[i], ada_prelim[i], rf_prelim[i]])

	x_test_dt = []
	for i in range(len(logreg_y)):
		x_test_dt.append([logreg_y[i], ada_y[i], rf_y[i]])

	dt = tree.DecisionTreeClassifier(max_depth=6, min_samples_split=4, min_samples_leaf=2, max_features=None)
	dt = dt.fit(info_for_dt, prelim_test_label)
	dt_preds = list(dt.predict(x_test_dt))

	assert(len(dt_preds) == len(y_test))
	print(dt_preds)

	accuracy = accuracy_score(y_test, dt_preds)
	prec = precision_score(y_test, dt_preds)
	recall = recall_score(y_test, dt_preds)

	return accuracy, prec, recall

def main():
	if len(sys.argv) != 3:
		print('ERROR: too many or too few arguments. Please re-run.')
		print('$python3 < robocall transcripts filename >   < non-robocall transcripts filename >')
		exit()

	robofilename , nonrobofilename = sys.argv[1] , sys.argv[2]

	# print(string.punctuation)
	giveaways = parsegiveaway('giveaways.txt')

	# get rid of the exit() to run the tree classifiers
	TRAIN_TEXT_FILE = 'text_traindata.csv'
	TEST_VOICE_FILE = 'full_testdata.csv'
	spam_ham_predict(TRAIN_TEXT_FILE, TEST_VOICE_FILE, giveaways)
	#exit()

	robofeat = extracttrain(giveaways,1, filename=robofilename)
	nonrobofeat = extracttrain(giveaways,0, filename=nonrobofilename)

	data = []
	for fv in robofeat:
		data.append(fv)
	for fv in nonrobofeat:
		data.append(fv)

	# predict just based on giveaway words
	max_accuracy = 0
	avg_accuracy = 0
	for i in range(0, 100):
		traindata, trainlabel, testfeatures, testlabels = get_random_data(data)
		giveaway_pred = [0]*len(testfeatures)
		for idx, t in enumerate(testfeatures):
			giveaway_pred[idx] = 0 if t[1] == 0 else 1
		num_correct = 0
		for i in range(len(giveaway_pred)):
			if giveaway_pred[i] == testlabels[i]:
				num_correct += 1
		accuracy = num_correct / len(testlabels)
		if accuracy > max_accuracy:
			max_accuracy = accuracy
		avg_accuracy += accuracy
	print('max giveaway accuracy after 100 tries:', max_accuracy)
	avg_accuracy /= 100
	print('average giveaway accuracy after 100 tries:', avg_accuracy)

	# default decision tree
	max_accuracy = 0
	avg_accuracy = 0
	avg_prec = 0
	avg_recall = 0
	for i in range(0, 100):
		traindata, trainlabel, testfeatures, testlabels = get_random_data(data)
		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(traindata, trainlabel)
		predictions = clf.predict(testfeatures)
		correct = 0
		for p , g in zip(predictions, testlabels):
			if p == g:
				correct += 1
		accuracy = correct/len(predictions)
		if accuracy > max_accuracy:
			max_accuracy = accuracy
		avg_accuracy += accuracy
		prec = precision_score(testlabels, predictions)
		recall = recall_score(testlabels, predictions)
		avg_prec += prec
		avg_recall += recall
	print('max default tree accuracy after 100 tries:', max_accuracy)
	avg_accuracy /= 100
	avg_prec /= 100
	avg_recall /= 100
	print('average default tree accuracy after 100 tries:', avg_accuracy)
	print('average default tree precision:', avg_prec)
	print('average dafault tree recall:', avg_recall)
	#exit()

	# individually tuned, bootstrapped trees
	print('beginning bootstrapped tree classifier')
	num_bootstrap = 5000
	traindata, trainlabel, testfeatures, testlabels = get_random_data(data)
	pred_dict = bootstrap_tree(traindata, trainlabel, testfeatures, num_bootstrap)
	print('assessing majority vote')
	# calculate majority vote from bootstrapped classifiers
	pred_list = [0] * len(testfeatures)
	print(testfeatures)
	for idx, item in enumerate(testfeatures):
		pred_list[idx] = 1 if pred_dict[tuple(item)] > num_bootstrap / 2 else 0
	num_correct = 0
	for i in range(len(pred_list)):
		if pred_list[i] == testlabels[i]:
			num_correct += 1
	accuracy = num_correct / len(testlabels)
	prec = precision_score(testlabels, pred_list)
	recall = recall_score(testlabels, pred_list)
	print('Bootstrap accuracy: ', accuracy)
	print('Bootstrap precision', prec)
	print('Bootstrap recall', recall)
	#exit()


	# Diverse Majority Vote and ensemble model
	maj_ac, maj_prec, maj_rec = 0, 0, 0
	ens_ac, ens_prec, ens_rec = 0, 0, 0
	for i in range(0, 100):
		traindata, trainlabel, testfeatures, testlabels = get_random_data(data)
		m_a, m_p, m_r = majority_vote(traindata, trainlabel, testfeatures, testlabels)
		e_a, e_p, e_r = ensemble(traindata, trainlabel, testfeatures, testlabels)
		maj_ac += m_a
		maj_prec += m_p
		maj_rec += m_r
		ens_ac += e_a
		ens_prec += e_p
		ens_rec += e_r
	print('average majority vote accuracy', maj_ac/100)
	print('average majority vote precision', maj_prec/100)
	print('average majority vote recall', maj_rec/100)
	print('average ensemble accuracy', ens_ac/100)
	print('average ensemble precision', ens_prec/100)
	print('average ensemble recall', ens_rec/100)

	# this part of the code trains on text message data
	train_lines = []
	with open(TRAIN_TEXT_FILE, 'r', encoding='ISO-8859-1') as tr:
		for line in tr.readlines():
			train_lines.append(line.strip())
	train_data = extracttrain(giveaways, 0, lines=train_lines)
	np.random.shuffle(train_data)

	# split training data into features and labels
	train_features, train_labels = [], []
	for fv in train_data:
		train_features.append(fv[:-1])
		train_labels.append(fv[-1])

	# prepare test data
	test_lines= []
	with open(TEST_VOICE_FILE, 'r', encoding='ISO-8859-1') as tf:
		for line in tf.readlines():
			test_lines.append(line.strip())
	test_data = extracttrain(giveaways, 1, lines=test_lines)
	test_features, test_labels = [], []
	for fv in test_data:
		test_features.append(fv[:-1])
		test_labels.append(fv[-1])

	# run the majority vote and ensemble model classifiers
	print('running majority vote on text-trained data')
	m_a, m_p, m_r = majority_vote(train_features, train_labels, test_features, test_labels)
	print('running ensemble model on text-trained data')
	e_a, e_p, e_r = ensemble(train_features, train_labels, test_features, test_labels)
	print('text-trained majority vote metrics:', m_a, m_p, m_r)
	print('text_trained ensemble metrics:', e_a, e_p, e_r)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()

