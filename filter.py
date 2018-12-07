#!/usr/bin/env python3
import sys
import string
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

giveaways = {}

def parsegiveaway(giveaway_filename):
	giveaways = set()
	file = open(giveaway_filename, 'r')
	for line in file:
		word = line[:-1] #strip '\n'
		giveaways.add(word)

	return giveaways


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

def get_random_data(data):
	np.random.shuffle(data)
	traindata , trainlabel = [] , []

	#testfeatures , testlabels = extracttest(testfilename)
	testfeatures = []
	testlabels = []
	i = 0
	for fv in data:
		if i < 25:
			traindata.append(fv[:-1])
			trainlabel.append(fv[-1])
		else:
			testfeatures.append(fv[:-1])
			testlabels.append(fv[-1])
		i += 1

	return traindata, trainlabel, testfeatures, testlabels

# predict spam/ham test messages
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

'''
Run ensemble model where the majority vote is taken from
all the different types of predictors
'''
def ensemble(x_train, y_train, x_test, y_test):
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
	print(rf_preds)

	# decision tree
	dt = tree.DecisionTreeClassifier(max_depth=6, min_samples_split=4, min_samples_leaf=2, max_features=None)
	dt = dt.fit(x_train, y_train)
	dt_preds = list(dt.predict(x_test))

	assert(len(logreg_preds) == len(ada_preds) == len(rf_preds) == len(dt_preds))
	aggregate_pred = [sum(x) for x in zip(logreg_preds, ada_preds, rf_preds, dt_preds)]
	print(aggregate_pred)

	final_pred = [0]*len(aggregate_pred)
	for i in range(len(final_pred)):
		if aggregate_pred[i] > 2:
			final_pred[i] = 1

	accuracy = accuracy_score(y_test, final_pred)
	print('ensemble accuracy', accuracy)

	print(final_pred)
	print(y_test)

def main():
	if len(sys.argv) != 4:
		print('ERROR: too many or too few arguments. Please re-run.')
		print('$python3 < robocall transcripts filename >   < non-robocall transcripts filename >')
		exit()

	robofilename , nonrobofilename , testfilename = sys.argv[1] , sys.argv[2] , sys.argv[3]

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
	print('max default tree accuracy after 100 tries:', max_accuracy)
	avg_accuracy /= 100
	print('average default tree accuracy after 100 tries:', avg_accuracy)
	#exit()

	'''
	# individually tuned, bootstrapped tres
	num_bootstrap = 5000
	traindata, trainlabel, testfeatures, testlabels = get_random_data(data)
	pred_dict = bootstrap_tree(traindata, trainlabel, testfeatures, num_bootstrap)
	# calculate majority vote from bootstrapped classifiers
	pred_list = [0] * len(testfeatures)
	print(testfeatures)
	for idx, data in enumerate(testfeatures):
		pred_list[idx] = 1 if pred_dict[tuple(data)] > num_bootstrap / 2 else 0
	num_correct = 0
	for i in range(len(pred_list)):
		if pred_list[i] == testlabels[i]:
			num_correct += 1
	accuracy = num_correct / len(testlabels)
	print('Majority vote accuracy: ', accuracy)
	'''

	# do an ensemble model
	traindata, trainlabel, testfeatures, testlabels = get_random_data(data)
	ensemble(traindata, trainlabel, testfeatures, testlabels)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()

