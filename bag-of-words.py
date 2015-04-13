# -*- coding:utf-8 -*-
__author__ = 'JasonTan'

import pandas as pd
from bs4 import BeautifulSoup as soup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

import re

import logging
# Display progress logs on stdout
#logging.basicConfig(filename='example.log',level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    filename='bag-of-words.log')

train = pd.read_csv("arabic.train.three.csv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("arabic.test.three.csv", header=0, delimiter="\t", quoting=3)
devel = pd.read_csv("arabic.devel.three.csv", header=0, delimiter="\t", quoting=3)

y_train = train["result"]
y_test = test["result"]
y_devel = devel["result"]

print y_train.shape

def review_to_words(raw_review):
    review_text = soup(raw_review)

    letters_only = re.sub("[^a-zA-Z]", " ", review_text.get_text())

    words = letters_only.lower().split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops ]

    return " ".join(meaningful_words)

num_reviews = train["text"].size

clean_train_reviews = []

for i in xrange(0, num_reviews):
    if i % 1000 == 0:
        logging.info("processing %d text..." % i)

    clean_train_reviews.append(review_to_words(train["text"][i]))


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.6, use_idf=True,
                            ngram_range=(1, 2), analyzer="word", tokenizer=None, preprocessor=None,
                            stop_words="english", max_features=4000)

logging.debug(vectorizer)
train_data_features = vectorizer.fit_transform(clean_train_reviews)




def get_features(data):
    num_reviews = len(data["text"])
    clean_test_reviews = []

    logging.info("clearning and parsing the test set reviews ...\n")
    for i in xrange(0, num_reviews):
        if i % 1000 == 0:
            logging.info("processing the %d reviews..." % i)
        clean_test_reviews.append(review_to_words(data["text"][i]))

    test_data_features = vectorizer.transform(clean_test_reviews)

    return test_data_features


test_data_features = get_features(test)
devel_data_features = get_features(devel)

logging.info("old train_data_features shape is ")
logging.info(train_data_features.shape)
#卡方选取特征
from sklearn.feature_selection import SelectKBest, chi2
ch2 = SelectKBest(chi2, k=2000)
train_data_features = ch2.fit_transform(train_data_features, y_train)
test_data_features = ch2.transform(test_data_features)
devel_data_features = ch2.transform(devel_data_features)


from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import cPickle

logging.info("save train data...")
dump_svmlight_file(train_data_features, y_train, open("feature/arabic.train.bagofwords.1000.libsvm",'wb'), zero_based=False)

logging.info("saving test data")
dump_svmlight_file(test_data_features, y_test, open("feature/arabic.test.bagofwords.1000.libsvm", 'wb'), zero_based=False)

logging.info("saving test data")
dump_svmlight_file(devel_data_features, y_devel, open("feature/arabic.devel.bagofwords.1000.libsvm", 'wb'), zero_based=False)