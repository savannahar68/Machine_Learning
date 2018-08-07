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