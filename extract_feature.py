# -*- coding: utf-8 -*-
__author__ = 'tan'

from nltk.stem import SnowballStemmer
from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from model import Question, Answer

english_stemmer = SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    '''
    抽词干 TfIdf 向量化
    '''
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (
            english_stemmer.stem(w) for w in analyzer(doc)
        )


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

            fea['gold'] = com.gold

            meta.append(fea)

        return meta


class ContentFeature(FeatureBase):
    '''
    'NumCommWords',  # 问题和答案单词重叠率
    'CosineSim',  # 问句和答案的余弦相似性
    'ComLen',   # c的长度
    '''
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
            fea['ComLen'] = len(com.body)
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
    'NumWhQ',
    'NumWh',
    'NumNEQ',
    'NumNE',
    'NumBigram',
    'NumTrigram',
    'NumFourgram',
    'NumFivegram'
    '''
    def extract_feature(self, question):
        print("processing question --> " + question.id)
        meta = []
        qbody = question.subject + " " + question.body
        numNN, numVB, numPronoun, numWh, numNE, numWords = self.get_num(qbody)
        qwords = word_tokenize(qbody)
        qpostag = [tag[1] for tag in pos_tag(qwords)]
        qbigram = self.ngram(qpostag, 2)
        qtrigram = self.ngram(qpostag, 3)
        qfourgram = self.ngram(qpostag, 4)
        qfivegram = self.ngram(qpostag, 5)

        for index, com in enumerate(question.answers):
            fea = {}
            cnumNN, cnumVB, cnumPronoun, cnumWh, cnumNE, cnumWords = self.get_num(com.body)
            fea['RatioNN'] = cnumNN*1.0 / cnumWords if cnumWords > 0 else 0
            fea['RatioVB'] = cnumVB*1.0 / cnumWords if cnumWords > 0 else 0
            fea['RatioPronoun'] = cnumPronoun*1.0 / cnumWords if cnumWords > 0 else 0
            fea['NumPronoun'] = cnumPronoun
            fea['NumWh'] = cnumWh
            fea['NumNE'] = cnumNE
            fea['NumNN'] = cnumNN
            fea['NumVB'] = cnumVB

            fea['NumPronounQ'] = numPronoun
            fea['NumWhQ'] = numWh
            fea['NumNEQ'] = numNE
            fea['NumNNQ'] = numNN
            fea['NumVBQ'] = numVB

            # print 'RatioNN', fea['RatioNN'], 'RatioVB', fea['RatioVB'], 'NumPronoun', fea['NumPronoun'],  'RatioPronoun', fea['RatioPronoun'], cnumPronoun, cnumWords


            cwords = word_tokenize(com.body)
            cpostag = [tag[1] for tag in pos_tag(cwords)]
            bigram = self.ngram(cpostag, 2)
            trigram = self.ngram(cpostag, 3)
            fourgram = self.ngram(cpostag, 4)
            fivegram = self.ngram(cpostag, 5)

            fea['NumBigram'] = self.com_ngrams(qbigram, bigram)
            fea['NumTrigram'] = self.com_ngrams(qtrigram, trigram)
            fea['NumFourgram'] = self.com_ngrams(qfourgram, fourgram)
            fea['NumFivegram'] = self.com_ngrams(qfivegram, fivegram)
            # print fea['NumBigram'], fea['NumTrigram'], qbigram, bigram


            meta.append(fea)
        return meta

    def get_num(self, text):
        numNN = 0
        numVB = 0
        numWh = 0
        numNE = 0
        numPronoun = 0
        numWords = 0
        sents = sent_tokenize(text)
        for sent in sents:
            words = word_tokenize(sent)
            numWords += len(words)
            try:
                tags = pos_tag(words)
            except UnicodeDecodeError, ee:
                print(sents)
                print(ee)
            nes = ne_chunk(tags, binary=True)
            for tag in tags:
                if tag[1].startswith('NN'):
                    numNN += 1
                elif tag[1].startswith('W'):
                    numWh += 1
                elif tag[1].startswith('VB'):
                    numVB += 1
                elif tag[1].startswith('PRP'):
                    numPronoun += 1

            for ne in nes.subtrees():
                from nltk import tree
                if isinstance(ne, tree.Tree):
                    if ne.label() == 'NE':
                        numNE += 1
        return numNN, numVB, numPronoun, numWh, numNE, numWords

    def ngram(self, postag, n):
        ngrams = []
        for inx in range(len(postag)):
            if inx + n < len(postag):
                ngrams.append(tuple(postag[inx:inx+n]))
            else:
                # ngrams.append(tuple(postag[inx:]))
                break
        return ngrams

    def com_ngrams(self, q_ngram, c_ngram):
        if len(q_ngram) > 0:
            return len([ng for ng in c_ngram if ng in q_ngram])*1.0 / len(q_ngram)
        return 0.

from config import BASEDIR
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
    #print("in computeTfidf ",questions)
    bodys = [q.body for q in train_questions]
    bodys.extend([c.body for q in train_questions for c in q.answers])
    if devel_questions is not None:
        bodys.extend([q.body for q in devel_questions])
        bodys.extend([c.body for q in devel_questions for c in q.answers])
    vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', decode_error='ignore')
    #print(bodys)
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
        featureObj = [klFeature()]
    else:
        featureObj = [ContentFeature(), NoContentFeature(), posFeature(), klFeature()]
        train_questions = cPickle.load(open(BASEDIR+os.sep+trans_train_questions_file_name))
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
from config import trans_train_feature_file_name, trans_train_questions_file_name
from config import trans_devel_feature_file_name, trans_devel_questions_file_name
from config import trans_test_feature_file_name, trans_test_questions_file_name

def save_feature():
    train = True
    devel = True
    test = True

    from config import BASEDIR
    if train:
        train_questions = cPickle.load(open(BASEDIR + os.sep + trans_train_questions_file_name, 'rb'))
        meta = prepare_sent_features(trans_train_feature_file_name, train_questions)
        cPickle.dump(meta, open(BASEDIR + os.sep + trans_train_feature_file_name, "wb"))
        save_as_libsvm(BASEDIR + os.sep + trans_train_feature_file_name, meta)

    if devel:
        devel_questions = cPickle.load(open(BASEDIR + os.sep + trans_devel_questions_file_name, 'rb'))
        meta = prepare_sent_features(trans_devel_feature_file_name, devel_questions)
        cPickle.dump(meta, open(BASEDIR + os.sep + trans_devel_feature_file_name, "wb"))
        save_as_libsvm(BASEDIR + os.sep + trans_devel_feature_file_name, meta)

    if test:
        test_questions = cPickle.load(open(BASEDIR + os.sep + trans_test_questions_file_name, 'rb'))
        meta = prepare_sent_features(trans_test_feature_file_name, test_questions)
        cPickle.dump(meta, open(BASEDIR + os.sep + trans_test_feature_file_name, "wb"))
        save_as_libsvm(BASEDIR + os.sep + trans_test_feature_file_name, meta)




if __name__ == "__main__":
    # f = NoContentFeature()
    # extract(f)
    save_feature()
    # save_label()