# -*- coding: utf-8 -*-
__author__ = 'tan'

import os
import logging

logname = "log"+os.sep+os.path.basename(__file__) + ".log"
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    filename=logname)

#配置java环境
java_path = "E:/Program Files/Java/jdk1.8.0_25/bin/java.exe"
os.environ['JAVAHOME'] = java_path

tagger_path = 'D:/research/stanford-postagger-full-2015-01-30/models/arabic.tagger'
jar_path = 'D:/research/stanford-postagger-full-2015-01-30/stanford-postagger.jar'


encoding = "utf-8"


from nltk.internals import java
import tempfile
from subprocess import PIPE

import sys
def POStag_sents(sentences):
    '''

    :param sentences:
    :return:
    '''
    if len(sentences.strip()) == 0 or sentences.strip() == "":
        return []

    _cmd = ['edu.stanford.nlp.tagger.maxent.MaxentTagger',
                '-model', tagger_path, '-tokenize', 'false']
    # Create a temporary input file
    _input_fh, _input_file_path = tempfile.mkstemp(text=True)

    _cmd.extend(['-encoding', encoding])
    _cmd.extend(['-textFile', _input_file_path])

    # Write the actual sentences to the temporary input file
    _input_fh = os.fdopen(_input_fh, 'wb')
    _input_fh.write(sentences.encode(encoding))
    _input_fh.close()

    # Run the tagger and get the output
    try:
        stanpos_output, _stderr = java(_cmd, classpath=jar_path,
                                                   stdout=PIPE, stderr=PIPE)
    except WindowsError:
        print(_cmd)
        print(sentences)
        sys.exit()
    stanpos_output = stanpos_output.decode(encoding)

    # Delete the temporary file
    os.unlink(_input_file_path)

    return parse_output(stanpos_output)

def parse_output(text):
    # Output the tagged sentences
    tagged_sentences = []
    for tagged_sentence in text.strip().split("\n"):
        sentence = []
        for tagged_word in tagged_sentence.strip().split():
            word_tags = tagged_word.strip().split("/")
            sentence.append((''.join(word_tags[:-1]), word_tags[-1]))
        tagged_sentences.extend(sentence)
    return tagged_sentences

#java -mx1g edu.stanford.nlp.international.arabic.process.ArabicSegmenter -loadClassifier data/arabic-segmenter-atb+bn+arztrain.ser.gz -textFile my_arabic_file.txt > my_arabic_file.txt.segmented


def word_segment(sentences):
    model_file = r"D:/research/stanford-segmenter-2015-01-30/data/arabic-segmenter-atb+bn+arztrain.ser.gz"
    _cmd = ["edu.stanford.nlp.international.arabic.process.ArabicSegmenter",
            "-loadClassifier", model_file]

    jar_path = "D:/research/stanford-segmenter-2015-01-30/stanford-segmenter-3.5.1.jar"
    # Create a temporary input file
    _input_fh, _input_file_path = tempfile.mkstemp(text=True)

    _cmd.extend(['-textFile', _input_file_path])

    # Write the actual sentences to the temporary input file
    _input_fh = os.fdopen(_input_fh, 'wb')
    _input_fh.write(sentences.encode(encoding))
    _input_fh.close()

    stanpos_output, _stderr = java(_cmd, classpath=jar_path, stdout=PIPE, stderr=PIPE)
    stanpos_output = stanpos_output.decode(encoding)

    # Delete the temporary file
    os.unlink(_input_file_path)

    return stanpos_output


if __name__ == "__main__":

    str = ur'يوجد أشخاص يشترون العملة العراقي ويحتفظون بها إلى أن ترتفع سعر البيع، فهل هذا حلال أم حرام؟ (تجارة العملة حرام أم حلال)؟'

    from nltk import word_tokenize, sent_tokenize
    tmp = word_tokenize(str)
    # print(tmp)
    print("tokenize ", len(tmp))
    tmp = sent_tokenize(str)
    print("sent ", len(tmp))
    print("str ", len(str))
    res = word_segment(str)
    print(word_segment(str+" "+str))
    print("word_segment is ")
    print(res)
    tmp = word_tokenize(res)

    def out(pos):
        for p in pos[0]:
            print p[0], p[1]

    out(POStag_sents(tmp))

    print(tmp)
    print(len(tmp))
    tmp = sent_tokenize(res)
    print(tmp)
    print(len(tmp))
    print(len(res))



    # segment(str)