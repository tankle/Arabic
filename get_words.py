# -*- coding: utf-8 -*-
__author__ = 'tan'

import cPickle
import os
from config import BASEDIR, trans_train_questions_file_name, trans_devel_questions_file_name
from config import trans_test_questions_file_name
from config import targets

train_questions = cPickle.load(open(BASEDIR + os.sep + trans_train_questions_file_name, 'rb'))
devel_questions = cPickle.load(open(BASEDIR+os.sep + trans_devel_questions_file_name, "rb"))
test_questions = cPickle.load(open(BASEDIR+os.sep + trans_test_questions_file_name, "rb"))

import re
def replace(text):
    # text = re.sub("\"", "\\\"", text)
    text = re.sub("\n", " ", text)
    text = re.sub("\t", " ", text)
    return text


def outfile(questins, outname):
    print "processing ", outname
    with open(outname, "wb") as f:
        f.write("\t".join(["id", "result", "text"]))
        f.write("\n")
        for ques in questins:
            text = ques.subject + " " + ques.body
            for com in ques.answers:
                cid = ques.id +"-"+com.id
                result = targets[com.gold]
                # result = com.gold
                text = com.body + " " + text
                f.write(cid+"\t"+str(result))
                f.write("\t\"" + replace(unicode(text.strip()).encode("utf-8")) + "\"")
                f.write("\n")


# outfile(train_questions, "arabic.train.three.csv")
# outfile(devel_questions, "arabic.devel.three.csv")
outfile(test_questions, "arabic.test.three.csv")
