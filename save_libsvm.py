# -*- coding: utf-8 -*-
__author__ = 'tan'


from config import feature_names, arabic_feature_names
import cPickle



def get_features(meta, qid, aid, flag):
    if flag is True:
        return tuple(meta[qid][aid][fn] for fn in arabic_feature_names)
    return tuple(meta[qid][aid][fn] for fn in feature_names)


# 输出时需要修改这个
two_class = False

def save_as_libsvm(feature_file_name, meta, flag=False):
    '''
    :param feature_file_name:
    :param meta:
    :param flag: 用来判断是English 还是Arabic的特征
    :return:
    '''
    from config import targets
    if two_class is True:
        outfile = open(feature_file_name + ".two.libsvm", 'wb')
    else:
        outfile = open(feature_file_name + ".three.libsvm", 'wb')
    print("save the feature file into svm! outname is " + outfile.name)
    for qid in range(len(meta)):
        for aid in range(len(meta[qid])):
            # print(qid, aid, meta[qid][aid])
            feas = get_features(meta, qid, aid, flag)
            target = targets[meta[qid][aid]['gold']]

            if two_class is True:
                if target == 1:
                    outfile.write('1')
                else:
                    outfile.write('0')
            else:
                outfile.write(str(target))
            for inx, fea in enumerate(feas):
                outfile.write(" " + str(inx+1) + ":" + str(fea))
            outfile.write("\n")
    outfile.close()


def save_tag(outname, questions):
    """
    根据questions得到最后的tag文件
    :param outname:
    :param questions:
    :return:
    """
    with open(outname, 'wb') as outfile:
      for ques in questions:
          for com in ques.answers:
              outfile.write(ques.id+"-"+com.id+"\n")


def merge_tag_result(tagname, resultname, outname):
    """
    合并tag 文件 和result文件
    得到最后的结果文件
    :param tagname:
    :param resultname:
    :param outname:
    :return:
    """
    f = open(BASEDIR+os.sep+tagname, 'rb')
    tags = f.readlines()
    f.close()

    f = open(BASEDIR+os.sep+resultname, 'rb')
    results = f.readlines()
    f.close()

    if len(tags) != len(results):
        print("The length of tags and results is not equal!!!")
        exit(0)
    from config import targets
    retarget = dict(zip(targets.values(), targets.keys()))
    with open(BASEDIR+os.sep+outname, 'wb') as f:
        for inx in xrange(len(tags)):
            f.write(tags[inx].strip() + "\t" + retarget[int(results[inx])]+"\n")


from utils import readFile
from config import BASEDIR
import os
from config import train_file_name, train_questions_file_name
from config import devel_file_name, devel_questions_file_name
from config import test_file_name, test_questions_file_name
from config import trans_train_file_name, trans_train_questions_file_name
from config import trans_devel_file_name, trans_devel_questions_file_name
from config import trans_test_file_name, trans_test_questions_file_name


def save_questions(file_name, outfile_name):
    print("processing file %s " % file_name)
    questions = readFile(file_name)
    cPickle.dump(questions, open(BASEDIR + os.sep + outfile_name, 'wb'))

def get_questions_dump():
    '''
    获取数据集的dump格式，
    只需要运行一次即可
    :return:
    '''
    save_questions(train_file_name, train_questions_file_name)
    save_questions(devel_file_name, devel_questions_file_name)
    save_questions(test_file_name, test_questions_file_name)
    # save_questions(trans_train_file_name, trans_train_questions_file_name)
    # save_questions(trans_devel_file_name, trans_devel_questions_file_name)
    # save_questions(trans_test_file_name, trans_test_questions_file_name)



from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from utils import merge_two_libsvm

def merge_two_feature_file(name1, name2, outname):
    name1 = "feature" + os.sep + name1
    name2 = "feature" + os.sep + name2
    print "processing ", name1, name2

    print("saving into {}".format(outname))

    #TODO 如果是加bag-of-words 特征，最好是加上这个参数n_features=4000,
    x_train, y_train = load_svmlight_file(name1, n_features=1000, zero_based=False)
    # x_train, y_train = load_svmlight_file(name1, zero_based=False)
    x_train_add, y_train_add = load_svmlight_file(name2, zero_based=False)

    new_x_train = merge_two_libsvm(x_train, x_train_add)
    dump_svmlight_file(new_x_train.toarray(), y_train_add, open("feature"+os.sep+outname, 'wb'), zero_based=False)

    # if sum(y_train == y_train_add) == len(y_train):
    #     new_x_train = merge_two_libsvm(x_train, x_train_add)
    #     dump_svmlight_file(new_x_train.toarray(), y_train, open("feature"+os.sep+outname, 'wb'), zero_based=False)
    # else:
    #     print "In the two file ", name1, name2, " the y label is not equal !!!"

def merge_feature():
    strone = "Arabic_%s_feature_file.dump.three.libsvm"
    strtwo = "trans_Arabic_%s_feature_file.dump.three.libsvm"

    train = "train"
    test = "test"
    devel = "devel"

    merge_two_feature_file(strone % train, strtwo % train, "train.all.libsvm")
    merge_two_feature_file(strone % devel, strtwo % devel, "devel.all.libsvm")
    merge_two_feature_file(strone % test, strtwo % test, "test.all.libsvm")

def merge_bagofwords_feature():
    # strone = "arabic.%s.bagofwords.three.libsvm"
    strone = "Arabic_NoTrans_%s_question.dump.coocurr.libsvm"
    strtwo = "%s.all.libsvm"

    train = "train"
    test = "test"
    devel = "devel"

    merge_two_feature_file(strone % train, strtwo % train, "train.coocurr.libsvm")
    merge_two_feature_file(strone % devel, strtwo % devel, "devel.coocurr.libsvm")
    merge_two_feature_file(strone % test, strtwo % test, "test.coocurr.libsvm")

def changeFeatureFile(questions, name):
    '''
    将特征文件变为learning to rank 文件格式
    :param questions:
    :param name:
    :return:
    '''
    name = BASEDIR + os.sep + name
    print("pricessing "+name)
    x_train, y_train = load_svmlight_file(name, zero_based=False)
    bad_idx = y_train == 3
    good_idx = y_train == 1
    pro_idx = y_train == 2

    x_train = x_train.toarray()

    y_train[bad_idx] = 0
    y_train[pro_idx] = 1
    y_train[good_idx] = 2

    print("savint into {}.rank".format(name))
    with open(name+".rank", "wb") as f:
        i = 0
        for ques in questions:
            for ans in ques.answers:
                f.write(str(y_train[i])+" ")
                f.write("qid:"+ques.id)
                for j in xrange(0, len(x_train[i])):
                    f.write(" "+str(j+1)+":"+str(x_train[i][j]))
                f.write(" #"+ques.id+"-"+ans.id)
                f.write("\n")
                i += 1



def changeFeatureFileAll():
    # train = "train.libsvm"
    # test = "test.libsvm"
    # devel = "devel.libsvm"

    train = "arabic.train.bagofwords.1000.libsvm"
    test = "arabic.test.bagofwords.1000.libsvm"
    devel = "arabic.devel.bagofwords.1000.libsvm"

    question = cPickle.load(open(BASEDIR+os.sep+train_questions_file_name , "rb"))
    changeFeatureFile(question, train)

    question = cPickle.load(open(BASEDIR+os.sep+test_questions_file_name, "rb"))
    changeFeatureFile(question, test)

    question = cPickle.load(open(BASEDIR+os.sep+devel_questions_file_name,"rb"))
    changeFeatureFile(question, devel)

if __name__ == "__main__":
    # get_questions_dump()

    # merge_feature()

    merge_bagofwords_feature()

    # changeFeatureFileAll()

    # from config import devel_feature_file_name, train_feature_file_name, BASEDIR
    # train_meta = cPickle.load(open(BASEDIR + os.sep + train_feature_file_name, 'rb'))
    # devel_meta = cPickle.load(open(BASEDIR + os.sep + devel_feature_file_name, 'rb'))
    # save_as_libsvm(BASEDIR + os.sep + train_feature_file_name, train_meta)
    # save_as_libsvm(BASEDIR + os.sep + devel_feature_file_name, devel_meta)

    # from config import devel_questions_file_name, tag_file
    # questions = cPickle.load(open(BASEDIR+os.sep+train_questions_file_name, 'rb'))
    # save_tag("arbic_train_tag", questions)

    # questions = cPickle.load(open(BASEDIR+os.sep+devel_questions_file_name, 'rb'))
    # save_tag(BASEDIR+os.sep+tag_file, questions)

    # merge_tag_result(tag_file, "arbic.result", "arbic_tag_result.txt")
