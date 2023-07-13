import re
import nltk.data
from nltk import wordpunct_tokenize

text = '''There are two ways of constructing a software design:
One way is to make it so simple that there are obviously no deficiencies and
the other way is to make it so complicated that there are no obvious deficiencies.'''
# — C.A.R. Hoare, The 1980 ACM Turing Award Lecture

# split into words by punctuations
# remove punctuations and all '-' words
RE = re.compile('[0-9a-z-]', re.I)
words = filter(lambda w: RE.search(w) and w.replace('-', ''), wordpunct_tokenize(text))

wordc = len(words)
charc = sum(len(w) for w in words)

sent = nltk.data.load('tokenizers/punkt/english.pickle')

sents = sent.tokenize(text)
sentc = len(sents)

print words
print charc, wordc, sentc
print 4.71 * charc / wordc + 0.5 * wordc / sentc - 21.43

# reference: https://pypi.python.org/pypi/textstat/

from textstat.textstat import textstat
if __name__ == '__main__':
    test_data = """Playing games has always been thought to be important to the development of well-balanced and creative children; however, what part, if any, they should play in the lives of adults has never been researched that deeply. I believe that playing games is every bit as important for adults as for children. Not only is taking time out to play games with our children and other adults valuable to building interpersonal relationships but is also a wonderful way to release built up tension."""

print textstat.flesch_reading_ease(test_data) #doesn't work when punctuation removed because it counts sentences
print textstat.smog_index(test_data)
print textstat.flesch_kincaid_grade(test_data)
print textstat.coleman_liau_index(test_data)
print textstat.automated_readability_index(test_data)
print textstat.dale_chall_readability_score(test_data) #this is the best one to use, according to academic research
print textstat.difficult_words(test_data)
print textstat.linsear_write_formula(test_data)
print textstat.gunning_fog(test_data)
print textstat.text_standard(test_data)

from __future__ import print_function
from time import time
import os
import random

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cross_validation import train_test_split
from textstat.textstat import textstat

import numpy as np

import pickle

validDocsDict = dict()

file_name = "TestDocsPub_kimtest.p"

validDocsDict = dict()
fileList1 = os.listdir("BioMedProcessed")

validDocsDict.update(pickle.load(open("BioMedProcessed/" + file_name, "rb")))

n_samples = len(validDocsDict.keys())
n_features = 1000
n_topics = 2
n_top_words = 30


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

from multiprocessing import Pool

print("Loading dataset...")
t0 = time()
#documents = []
readability = []

labels = []
concLengthTotal = 0
discLengthTotal = 0
#concCount = 0
#discCount = 0

def f(k):
    for k in validDocsDict.keys():
        if k.startswith("conclusion"):
            labels.append(0)
            #documents.append(validDocsDict[k])
            readability.append(textstat.dale_chall_readability_score(validDocsDict[k]))
            #concCount += 1
            #concLengthTotal += len(validDocsDict[k].split(' '))
        elif k.startswith("discussion"):
            labels.append(1)
            #documents.append(validDocsDict[k])
            readability.append(textstat.dale_chall_readability_score(validDocsDict[k]))
            #discCount += 1
            #discLengthTotal += len(validDocsDict[k].split(' '))
            
po = Pool(6)
results = [po.apply_async(f, args = (k,)) for k in validDocsDict.keys()]
po.close()
po.join()
output = [p.get() for p in results]

#print(len(documents))
#print(concLengthTotal * 1.0/ concCount)
#print(discLengthTotal * 1.0/ discCount)
print(len(readability))
#print(concCount + discCount)

train, test, labelsTrain, labelsTest = train_test_split(readability, labels, test_size = 0.1)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)
numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())
numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.linear_model import SGDClassifier

classifier = SGDClassifier()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(train, labelsTrain)

classResults = classifier.predict(test)

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.ensemble import BaggingClassifier

classifier = BaggingClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.ensemble import ExtraTreesClassifier

classifier = ExtraTreesClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier()

classifier.fit(train.toarray(), labelsTrain)

classResults = classifier.predict(test.toarray())

numRight = 0

for item in range(len(classResults)):
    if classResults[item] == labelsTest[item]:
        numRight += 1

print(str(numRight * 1.0 / len(classResults) * 1.0))

