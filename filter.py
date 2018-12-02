#!/usr/bin/env python3
import sys
import string
import random
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

giveaways = {}

def parsegiveaway(giveaway_filename):
	giveaways = set()
	file = open(giveaway_filename, 'r')
	for line in file:
		word = line[:-1] #strip '\n'
		giveaways.add(word)

	return giveaways


def parse(line):
	global giveaways

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


def extracttrain(filename, giveaways, label):

	features = []
	file = open(filename, 'r')

	for line in file:
		line = (line[:-1]).lower()
		wordCount , signalwordCount , umCount , punctuationCount = parse(line)
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

def main():

	if len(sys.argv) != 4:
		print('ERROR: too many or too few arguments. Please re-run.')
		print('$python3 < robocall transcripts filename >   < non-robocall transcripts filename >')
		exit()

	robofilename , nonrobofilename , testfilename = sys.argv[1] , sys.argv[2] , sys.argv[3]

	# print(string.punctuation)
	giveaways = parsegiveaway('giveaways.txt')
	robofeat = extracttrain(robofilename,giveaways,1)
	nonrobofeat = extracttrain(nonrobofilename,giveaways,0)

	data = []
	for fv in robofeat:
		data.append(fv)
	for fv in nonrobofeat:
		data.append(fv)

	random.shuffle(data)
	traindata , trainlabel = [] , []

	testfeatures , testlabels = extracttest(testfilename)
	i = 0
	for fv in data:
		if i < 40:
			traindata.append(fv[:-1])
			trainlabel.append(fv[-1])
		else:
			testfeatures.append(fv[:-1])
			testlabels.append(fv[-1])
		i += 1

	#Predict
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(traindata, trainlabel)
	predictions = clf.predict(testfeatures)
	correct = 0
	for p , g in zip(predictions, testlabels):
		if p == g:
			correct += 1
	print('Accuray: ' + str(correct/len(predictions)))


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()

