# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = 'tan'

import os
import logging

logname = "log" +os.sep + os.path.basename(__file__) + ".log"
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    filename=logname)

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause



from pprint import pprint
from time import time


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

import pandas as pd
train = pd.read_csv("arabic.train.three.csv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("arabic.test.three.csv", header=0, delimiter="\t", quoting=3)
devel = pd.read_csv("arabic.devel.three.csv", header=0, delimiter="\t", quoting=3)

y_train = train["result"]
y_test = test["result"]
y_devel = devel["result"]

###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.75, ngram_range=(1,2), max_features=5000)),
    ('tfidf', TfidfTransformer(norm='l1')),
    ('clf', LinearSVC(C=10)),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    # 'vect__max_df': ([0.75]),
    # 'vect__max_features': (None, 4000, 5000, 6000),
    # 'vect__ngram_range': ((1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__sublinear_tf': (True, False),
    # 'tfidf__norm': ('l1'),
    # 'clf__alpha': (0.00001, 0.000001),
    # 'clf__C': (10, 50, 100),
    # 'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(train["text"], y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    clf = grid_search.best_estimator_