#-*- coding: utf-8 -*-
__author__ = 'tan'


import numpy as np
import cPickle
import os

base_path = ur"""D:\百度云同步盘\Research\semeval2015\task3\arabic\semeval2015-task3-arabic-data\datasets"""


BASEDIR = os.path.split(os.path.realpath(__file__))[0] + os.sep + "feature"

#翻译后文件
trans_train_file_name = base_path + os.sep + "trans_QA-Arabic-train.xml"
trans_train_feature_file_name = "trans_Arabic_train_feature_file.dump"
trans_train_questions_file_name = "trans_Arabic_train_question.dump"

trans_devel_file_name = base_path + os.sep + "trans_QA-Arabic-dev.xml"
trans_devel_feature_file_name = "trans_Arabic_devel_feature_file.dump"
trans_devel_questions_file_name = "trans_Arabic_devel_question.dump"

trans_test_file_name = base_path + os.sep + "trans_test_task3_Arabic_new.xml"
trans_test_feature_file_name = "trans_Arabic_test_feature_file.dump"
trans_test_questions_file_name = "trans_Arabic_test_question.dump"


#没有翻译的原始文件
devel_file_name = base_path + os.sep + "QA-Arabic-dev.xml"
devel_feature_file_name = "Arabic_devel_feature_file.dump"
devel_questions_file_name = "Arabic_devel_question.dump"

train_file_name = base_path + os.sep + "QA-Arabic-train.xml"
train_feature_file_name = "Arabic_train_feature_file.dump"
train_questions_file_name = "Arabic_train_question.dump"

test_file_name = base_path + os.sep + "test_task3_Arabic_new.xml"
test_feature_file_name = "Arabic_test_feature_file.dump"
test_questions_file_name = "Arabic_test_question.dump"



tag_file = "arbic_devel_tag.txt"


update_feature = False

targets = {'direct': 1, 'related': 2, 'irrelevant': 3}

#抽取english的词特征数组
feature_names = np.array((
    'NumTextTokens',  # 词的个数
    'MaxWordsInSent',  # 句子中最长的单词长度
    'AvgSentLen',  # 句子的平均长度
    'NumSent',  # 句子的个数
    'AvgWordLen',  # 单词的平均长度
    'NumWordQSubject',  #
    'NumWordQBody',

    'NumCommWords',  # 问题和答案单词重叠率
    'CosineSim',  # 问句和答案的余弦相似性
    'ComLen',   # c的长度
    
    'RatioNN',  # NN的比例
    'RatioVB',  # VB的比例
    'RatioPronoun',  # 代词的比例
    'NumNN',    # NN的个数
    'NumVB',    # VB的个数
    'NumNNQ',   # NN在question的个数
    'NumVBQ',   # VB在question的个数
    'NumPronoun',   # 代词的个数
    'NumPronounQ',  # q中代词的个数
    'NumWhQ',   # q中wh类型的个数
    'NumWh',    # wh类型的个数
    'NumNEQ',   # q中NE的个数
    'NumNE',    # NE的个数
    'NumBigram',    # Q和C相同bigram的个数
    'NumTrigram',   # Q和C相同Trigram的个数
    'NumFourgram',  # Q和C相同的Fourgram个数
    'NumFivegram',  # Q和C相同Fivegram个数
    

    'klDis',  # KL距离
    #'gold' #这是用来保存类别的

))

#抽取的arabic词特征数组
arabic_feature_names = np.array((
    'NumTextTokens',  # 词的个数
    'MaxWordsInSent',  # 句子中最长的单词长度
    'AvgSentLen',  # 句子的平均长度
    'NumSent',  # 句子的个数
    'NumWordQSubject',  #
    'NumWordQBody',
    'ComLen',   # c的长度

    'NumCommWords',  # 问题和答案单词重叠率
    'CosineSim',  # 问句和答案的余弦相似性

    'klDis',

    'RatioNN',  # NN的比例
    'RatioVB',  # VB的比例
    'RatioPronoun',  # 代词的比例
    'NumNN',    # NN的个数
    'NumVB',    # VB的个数
    'NumNNQ',   # NN在question的个数
    'NumVBQ',   # VB在question的个数
    'NumPronoun',   # 代词的个数
    'NumPronounQ',  # q中代词的个数
    'NumBigram',    # Q和C相同bigram的个数
    'NumTrigram',   # Q和C相同Trigram的个数
    'NumFourgram',  # Q和C相同的Fourgram个数
    'NumFivegram',  # Q和C相同Fivegram个数
    #'gold' #这是用来保存类别的

))

re_repl = {
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bur\b": "you are",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bdont\b": "do not",
    r"\bdoesnt\b": "does not",
    r"\bdidnt\b": "did not",
    r"\bhasnt\b": "has not",
    r"\bhavent\b": "have not",
    r"\bhadnt\b": "had not",
    r"\bwouldnt\b": "would not",
    r"\bcannot\b": "can not",
    r"\bplz\b": "please",
    r"\bBTW\b": "by the way"
    }