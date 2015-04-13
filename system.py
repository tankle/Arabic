# -*- coding: utf-8 -*-
__author__ = 'tan'

import os
import logging

logname = "log" + os.sep+ os.path.basename(__file__) + ".log"
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    filename=logname)

from sklearn.datasets import load_svmlight_file, dump_svmlight_file

# def test_merge():
#     name1 = "feature"+os.sep+"train.bagofwords.four.libsvm"
#     name2 = "feature"+os.sep+"train_feature_file.dump.four.libsvm"
#
#
#
#     x_train, y_train = load_svmlight_file(name1, n_features=4000, zero_based=False)
#     x_train_add, y_train_add = load_svmlight_file(name2, zero_based=False)
#
#     from utils import merge_two_libsvm
#     if sum(y_train == y_train_add) == len(y_train):
#         new_x_train = merge_two_libsvm(x_train, x_train_add)
#         print new_x_train.shape


from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


def report(y_true, y_pred):
    logging.debug("Good VS Bad VS Pro")
    # logging.debug(classification_report(y_true, y_pred))
    logging.debug(confusion_matrix(y_true, y_pred))
    # logging.debug("micro f1 score %f " % f1_score(y_true, y_pred, average="micro"))
    logging.debug("macro f1 score %f " % f1_score(y_true, y_pred, average="macro"))
    print("macro f1 score %f " % f1_score(y_true, y_pred, average="macro"))
    logging.debug("accuracy score %f " % accuracy_score(y_true, y_pred))


def test(trainname, testname):
    print "running... "
    logging.debug("\n\n\n")
    logging.debug(trainname)
    logging.debug(testname)
    x_train, y_train = load_svmlight_file("feature"+os.sep+trainname, zero_based=False)
    x_test, y_test = load_svmlight_file("feature"+os.sep+testname, zero_based=False)

    # clf = LinearSVC(dual=False, C=126)
    # clf = AdaBoostClassifier(n_estimators=100)
    clf = RandomForestClassifier(n_estimators=2000, n_jobs=-1, max_depth=10)
    logging.debug(clf)
    clf.fit(x_train.toarray(), y_train)
    pred = clf.predict(x_test.toarray())

    report(y_test, pred)

from time import time
from sklearn.grid_search import GridSearchCV
# Utility function to report best scores
import numpy as np
from operator import itemgetter


def reportSearch(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def searchParameter(x_train, y_train, x_test, y_test, clf, param_grid):
    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, scoring="accuracy")
    start = time()
    grid_search.fit(x_train, y_train)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.grid_scores_)))
    reportSearch(grid_search.grid_scores_)

    y_pred = grid_search.predict(x_test)
    report(y_test, y_pred)


from sklearn.linear_model import LogisticRegression

def search(trainname, testname):
    print("searching now...")
    x_train, y_train = load_svmlight_file("feature"+os.sep+trainname, zero_based=False)
    x_test, y_test = load_svmlight_file("feature"+os.sep+testname, zero_based=False)
    # clf = GradientBoostingClassifier()
    # # use a full grid over all parameters
    # param_grid = {"max_depth": [3, 6, 9, None],
    #               "max_features": [1, 3, 10],
    #               "min_samples_split": [1, 3, 10],
    #               "min_samples_leaf": [1, 3, 10],
    #               "bootstrap": [True, False],
    #               "criterion": ["gini", "entropy"],
    #               "n_estimators": [500, 1000, 2000]
    #               }

    clf = LogisticRegression()
    param_grid = {"penalty": ['l2', 'l1'],
                  "tol": [1e-4, 0.01, 0.1, 1],
                  "C": [0.1, 1.0, 10, 50, 100],
                }

    # clf = RandomForestClassifier()
    # param_grid = {"n_estimators": [100, 400],
    #               #"criterion": ["gini", "entropy"],
    #               "max_depth": [4, 5],
    #               "min_samples_split": [3, 5, 7],
    #               "min_samples_leaf": [1, 3, 5, 7],
    #               #"max_features": ["auto", "log2", None],
    #               #"max_leaf_nodes": [None, 10, 50],
    #               #"bootstrap": [True, False],
    #               #"oob_score": [False, True]
    #               }

    # clf = LinearSVC(dual=False)
    # param_grid = {"penalty": ['l2'],
    #               "loss": ['l2'],
    #               "tol": [1e-4, 0.01, 0.1],
    #               "C": [1.0, 10, 100, 500, 1000]}

    searchParameter(x_train.toarray(), y_train, x_test.toarray(), y_test, clf, param_grid)




def system_one(trainname, testname):
    print "running... "
    logging.debug("\n\n\n")
    logging.debug(trainname)
    logging.debug(testname)
    x_train, y_train = load_svmlight_file("feature"+os.sep+trainname, zero_based=False)
    x_test, y_test = load_svmlight_file("feature"+os.sep+testname, zero_based=False)

    good_idx = y_train == 1
    pro_idx = y_train == 2

    from copy import copy
    new_y_train = copy(y_train)
    new_y_train[pro_idx] = 1

    clf = LinearSVC(dual=False, C=0.1)
    # clf = AdaBoostClassifier(n_estimators=50)
    # clf = RandomForestClassifier(n_estimators=100)
    logging.debug(clf)
    clf.fit(x_train.toarray(), new_y_train)
    badpred = clf.predict(x_test.toarray())
    # logging.debug(clf.predict_proba(x_test.toarray()))


    new_x_train = x_train[good_idx+pro_idx]
    new_y_train = y_train[good_idx+pro_idx]

    # clf = LinearSVC(dual=False, C=128)
    clf = AdaBoostClassifier(n_estimators=50)
    logging.debug(clf)
    clf.fit(new_x_train.toarray(), new_y_train)
    pred = clf.predict(x_test.toarray())
    logging.debug(clf.predict_proba(x_test.toarray()))

    bad_idx = badpred == 3
    pred[bad_idx] = 3

    report(y_test, pred)


def system_three(trainname, testname):
    '''
    基于随机森林
    :return:
    '''
    print "system_three running... "
    logging.debug("\n\n\n")
    logging.debug(trainname)
    logging.debug(testname)
    x_train, y_train = load_svmlight_file("feature"+os.sep+trainname, zero_based=False)
    x_test, y_test = load_svmlight_file("feature"+os.sep+testname, zero_based=False)

    # clf = LinearSVC(dual=False, C=126)
    # clf = AdaBoostClassifier(n_estimators=50)
    clf = RandomForestClassifier(n_estimators=300, max_leaf_nodes=15)
    logging.debug(clf)
    clf.fit(x_train.toarray(), y_train)
    badpred = clf.predict(x_test.toarray())
    logging.debug(clf.predict_proba(x_test.toarray()))

    pass

def test_all_feature():
    trainname = "train.coocurr.libsvm"
    testname = "test.coocurr.libsvm"
    search(trainname, testname)
    # test(trainname, testname)
    # system_one(trainname, testname)

    # trainname = "trans_Arabic_train_feature_file.dump.three.libsvm"
    # testname = "trans_Arabic_test_feature_file.dump.three.libsvm"
    # test(trainname, testname)
    # # system_one(trainname, testname)
    #
    # trainname = "Arabic_train_feature_file.dump.three.libsvm"
    # testname = "Arabic_test_feature_file.dump.three.libsvm"
    # test(trainname, testname)
    # # system_one(trainname, testname)
    #
    # trainname = "train.all.libsvm"
    # testname = "test.all.libsvm"
    # test(trainname, testname)
    # system_one(trainname, testname)

    # trainname = "train.libsvm"
    # testname = "test.libsvm"
    # test(trainname, testname)
    # system_one(trainname, testname)


def system_rank():

    from getResult import get_result

    # inname = "feature" + os.sep + "pro.result.txt"
    inname = "feature" + os.sep + "8.score.txt"

    get_result(inname)

    logging.debug("\n\n\n")
    logging.debug(inname)
    logging.debug("using ranking method!!!")
    resultname = "tmp.result.txt"
    f = open(resultname, "rb")
    lines = f.readlines()
    f.close()
    y_pred = []
    for l in lines:
        y_pred.append(int(l))

    testname = "Arabic_test_feature_file.dump.three.libsvm"
    x_test, y_test = load_svmlight_file("feature"+os.sep+testname, zero_based=False)

    if len(y_test) != len(y_pred):
        print("the length is not equal")
        exit(0)
    report(y_test, y_pred)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
import cPickle
from config import BASEDIR, test_questions_file_name

def system_pro():
    train_name = "feature/train.all.libsvm"
    test_name = "feature/test.all.libsvm"
    x_train, y_train = load_svmlight_file(train_name, zero_based=False)
    x_test, y_test = load_svmlight_file(test_name, zero_based=False)

    pro_idx = y_train == 2
    bad_idx = y_train == 3
    y_train[pro_idx] = 1
    y_train[bad_idx] = 0

    clf = RandomForestClassifier(n_estimators=500)
    # clf = DecisionTreeClassifier()
    # clf = GradientBoostingClassifier(n_estimators=500)
    # clf = SVC(probability=True)
    clf.fit(x_train.toarray(), y_train)
    pred_pro = clf.predict_proba(x_test.toarray())
    # pred_pro = clf.predict(x_test.toarray())

    print(len(pred_pro))
    print(len(y_test))
    questions = cPickle.load(open(BASEDIR+os.sep+test_questions_file_name, "rb"))
    f = open("feature/pro.result.txt", "wb")
    i = 0
    for ques in questions:
        for ans in ques.answers:
            f.write(ques.id)
            # print(pred_pro[i])
            f.write("\t1\t{}".format(pred_pro[i][1]))
            f.write("\n")
            i += 1

    f.close()



if __name__ == "__main__":
    test_all_feature()
    #
    # system_pro()
    # system_rank()

