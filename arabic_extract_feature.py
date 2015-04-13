# -*- coding: utf-8 -*-
__author__ = 'tan'

from nltk.stem import SnowballStemmer
from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from model import Question, Answer

from postagger import POStag_sents, word_segment


class FeatureBase:
    def __init__(self):
        """
        特征抽取基函数

        """
        pass

    def extract_feature(self, question):
        pass




class NoContentFeature(FeatureBase):
    def extract_feature(self, question):
        print('In NoContentFeature.extract_feature --> ' + question.id)
        meta = []
        num_words_subject = len(word_tokenize(question.subject))
        num_words_body = len(word_tokenize(question.body))
        for index, com in enumerate(question.answers):
            fea = {}
            text = com.body
            sent_lens = [len(word_tokenize(sent)) for sent in sent_tokenize(text)]
            fea['NumTextTokens'] = sum(sent_lens)
            fea['MaxWordsInSent'] = max(sent_lens) if len(sent_lens) > 0 else 0
            fea['NumSent'] = len(sent_lens)
            fea['AvgSentLen'] = np.mean(sent_lens) if len(sent_lens) > 0 else 0
            words = word_tokenize(text)
            fea['AvgWordLen'] = np.mean([len(w) for w in words]) if len(words) > 0 else 0
            fea['NumWordQSubject'] = num_words_subject
            fea['NumWordQBody'] = num_words_body
            fea['ComLen'] = len(word_tokenize(com.body))

            fea['gold'] = com.gold

            meta.append(fea)

        return meta


class ContentFeature(FeatureBase):

    def setVectorizer(self, vectorizer):
        self.vectorizer = vectorizer

    def extract_feature(self, question):
        print('In ContentFeature.extract_feature --> ' + question.id)
        meta = []
        qbody = question.body
        qwords = word_tokenize(qbody)
        qvec = self.vectorizer.transform([qbody])
        for index, com in enumerate(question.answers):
            fea = {}
            commwords = len([w for w in word_tokenize(com.body) if w in qwords])
            fea['NumCommWords'] = commwords*1.0 / len(qwords)
            cvec = self.vectorizer.transform([com.body])
            fea['CosineSim'] = cosine_distances(qvec, cvec)[0][0]

            meta.append(fea)
        return meta


class posFeature(FeatureBase):
    '''
    'RatioNN',
    'RatioVB',
    'RatioPronoun',
    'NumNN',
    'NumVB',
    'NumNNQ',
    'NumVBQ',
    'NumPronoun',
    'NumPronounQ',
    # 'NumWhQ',
    # 'NumWh',
    # 'NumNEQ',
    # 'NumNE',
    'NumBigram',
    'NumTrigram',
    'NumFourgram',
    'NumFivegram'
    '''
    def extract_feature(self, question):
        print("in posFeature  --> " + question.id)
        meta = []
        qbody = question.subject + " " + question.body
        numNN, numVB, numPronoun, numWords = self.get_num(qbody)
        # qwords = word_tokenize(qbody)
        qpostag = [tag[1] for tag in POStag_sents(qbody)]
        qbigram = self.ngram(qpostag, 2)
        qtrigram = self.ngram(qpostag, 3)
        qfourgram = self.ngram(qpostag, 4)
        qfivegram = self.ngram(qpostag, 5)

        for index, com in enumerate(question.answers):
            fea = {}
            cnumNN, cnumVB, cnumPronoun, cnumWords = self.get_num(com.body)
            fea['RatioNN'] = cnumNN*1.0 / cnumWords if cnumWords > 0 else 0
            fea['RatioVB'] = cnumVB*1.0 / cnumWords if cnumWords > 0 else 0
            fea['RatioPronoun'] = cnumPronoun*1.0 / cnumWords if cnumWords > 0 else 0

            fea['NumPronoun'] = cnumPronoun
            fea['NumNN'] = cnumNN
            fea['NumVB'] = cnumVB

            fea['NumPronounQ'] = numPronoun
            fea['NumNNQ'] = numNN
            fea['NumVBQ'] = numVB

            cpostag = [tag[1] for tag in POStag_sents(com.body)]
            bigram = self.ngram(cpostag, 2)
            trigram = self.ngram(cpostag, 3)
            fourgram = self.ngram(cpostag, 4)
            fivegram = self.ngram(cpostag, 5)

            fea['NumBigram'] = self.com_ngrams(qbigram, bigram)
            fea['NumTrigram'] = self.com_ngrams(qtrigram, trigram)
            fea['NumFourgram'] = self.com_ngrams(qfourgram, fourgram)
            fea['NumFivegram'] = self.com_ngrams(qfivegram, fivegram)

            meta.append(fea)
        return meta

    def get_num(self, text):
        numNN = 0
        numVB = 0
        numPronoun = 0
        tags = POStag_sents(text)
        numWords = len(tags)
        # print(tags)
        for tag in tags:
            if tag[1].startswith('NN'):
                numNN += 1
            elif tag[1].startswith('VB'):
                numVB += 1
            elif tag[1].startswith('PRP'):
                numPronoun += 1

        return numNN, numVB, numPronoun, numWords

    def ngram(self, postag, n):
        ngrams = []
        for inx in range(len(postag)):
            if inx + n < len(postag):
                ngrams.append(tuple(postag[inx:inx+n]))
            else:
                ngrams.append(tuple(postag[inx:]))
                break
        return ngrams

    def com_ngrams(self, q_ngram, c_ngram):
        if len(q_ngram) > 0:
            return len([ng for ng in c_ngram if ng in q_ngram])*1.0 / len(q_ngram)
        return 0.

from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy

class klFeature(FeatureBase):
    """
    'klDis',
    """
    def extract_feature(self, question):
        print('In klFeature.extract_feature --> ' + question.id)
        meta = []
        for index, com in enumerate(question.answers):
            fea = {}
            kl = self.kl_divergence(question.subject + " " + question.body, com.body)
            fea['klDis'] = kl
            meta.append(fea)
        return meta

    def kl_divergence(self, question, answer):
        vectorizer = CountVectorizer(min_df=1)
        x = vectorizer.fit_transform([question, answer])
        qx = x.toarray()[0]
        px = x.toarray()[1]
        cominx = [inx for inx, item in enumerate(qx) if item > 0 and px[inx] > 0]
        kl = entropy(px[cominx], qx[cominx])
        return kl

import os
import cPickle

def computeTfidf(train_questions, devel_questions=None):
    bodys = [q.body for q in train_questions]
    bodys.extend([c.body for q in train_questions for c in q.answers])
    if devel_questions is not None:
        bodys.extend([q.body for q in devel_questions])
        bodys.extend([c.body for q in devel_questions for c in q.answers])
    vectorizer = TfidfVectorizer(min_df=1, decode_error='ignore')
    vectorizer.fit_transform(bodys)
    return vectorizer

def prepare_sent_features(feature_file_name, questions):
    """
    计算特征
    :param feature_file_name:
    :param questions:
    :return:
    """
    flag = False
    if os.path.isfile(feature_file_name):
        meta = cPickle.load(open(feature_file_name, 'rb'))
        flag = True
    else:
        meta = [0] * len(questions)
        flag = False

    ##这里需要修改
    #TODO
    from config import update_feature
    if update_feature:
        featureObj = [posFeature()]
    else:
        # featureObj = [posFeature(), ContentFeature(), NoContentFeature(), klFeature()]
        featureObj = [ContentFeature(), posFeature(), NoContentFeature(), klFeature()]
        train_questions = cPickle.load(open(BASEDIR+os.sep+train_questions_file_name))
        vectorizer = computeTfidf(train_questions)
        featureObj[0].setVectorizer(vectorizer)

    for feaobj in featureObj:
        for index, ques in enumerate(questions):
            if flag is False and meta[index] == 0:
                meta[index] = [0] * len(ques.answers)

            fea = feaobj.extract_feature(ques)
            for j, fe in enumerate(fea):
                if flag is False and meta[index][j] == 0:
                    meta[index][j] = {}
                meta[index][j].update(fe)

    return meta

from save_libsvm import save_as_libsvm
from config import devel_questions_file_name, devel_feature_file_name
from config import train_feature_file_name, train_questions_file_name
from config import test_feature_file_name, test_questions_file_name


from config import BASEDIR

def save_feature():

    train = True
    devel = True
    test = True

    if train:
        train_questions = cPickle.load(open(BASEDIR + os.sep + train_questions_file_name, 'rb'))
        meta = prepare_sent_features(train_feature_file_name, train_questions)
        cPickle.dump(meta, open(BASEDIR + os.sep + train_feature_file_name, "wb"))
        save_as_libsvm(BASEDIR + os.sep + train_feature_file_name, meta, True)

    if devel:
        devel_questions = cPickle.load(open(BASEDIR + os.sep + devel_questions_file_name, 'rb'))
        meta = prepare_sent_features(devel_feature_file_name, devel_questions)
        cPickle.dump(meta, open(BASEDIR + os.sep + devel_feature_file_name, "wb"))
        save_as_libsvm(BASEDIR + os.sep + devel_feature_file_name, meta, True)

    if test:
        test_questions = cPickle.load(open(BASEDIR + os.sep + test_questions_file_name, 'rb'))
        meta = prepare_sent_features(test_feature_file_name, test_questions)
        cPickle.dump(meta, open(BASEDIR + os.sep + test_feature_file_name, "wb"))
        save_as_libsvm(BASEDIR + os.sep + test_feature_file_name, meta, True)




if __name__ == "__main__":
    # f = NoContentFeature()
    # extract(f)
    save_feature()
    # save_label()