'''
We will do sentiment analysis using the same model, where the data is not in the format in which we want
We'll import nltk and download all the packages

Sudo pip install nltk
Then go to Python3

import nltk
nltk.download()
d #this will download all the packages
'''
#the following things are for preprocessing the data
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer #just handles the grammar(run,running,ran etc removes everything and makes them same)
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000 #run this on GPU else laptop will crash as too much ram needed

def create_lexicon(pos, neg):
	lexicon = []
	for fil in [pos,neg]:
		with open(fil, 'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = word_tokenize(l)
				lexicon += list(all_words)
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon) #this will give a dictionary of words and occurence
	l2 = []
	for w in w_counts:
		if 1000 > w_counts[w] > 50 :
			l2.append(w)
	print(len(l2))		
	return l2

def sample_handling(sample, lexicon, classification):
	featureset = []
	'''
		[
			[0 1 0 0 0], [1 0] #1 0 is positive sentiment samepl
			[0 1 1 0 0],
			[]
		]
	'''
	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features)
			featureset.append([features, classification])
	return featureset
	
def create_featuresets_and_labels(pos, neg, test_size = 0.1):
	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling('pos.txt', lexicon, [1,0])		
	features += sample_handling('neg.txt', lexicon, [0,1])
	random.shuffle(features)

	features = np.array(features)	
	testing_size = int(test_size*len(features))
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][:-testing_size])
	test_y = list(features[:,1][:-testing_size])

	return train_x,train_y,test_x,test_y


if __name__ == '__main__':

	train_x,train_y,test_x,test_y = create_featuresets_and_labels('pos.txt', 'neg.txt')
	with open('sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y], f)










