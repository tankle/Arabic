# -*- coding: utf-8 -*-
__author__ = 'tan'

import os
import logging

logname = os.path.basename(__file__) + ".log"
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    filename=logname)


java_path = "E:/Program Files/Java/jdk1.8.0_25/bin/java.exe"
os.environ['JAVAHOME'] = java_path

from nltk.tag.stanford import POSTagger


tagger_path = 'D:/research/stanford-postagger-full-2015-01-30/models/arabic.tagger'

jar_path = 'D:/research/stanford-postagger-full-2015-01-30/stanford-postagger.jar'
st = POSTagger(tagger_path, jar_path, encoding="utf-8")

st._SEPARATOR = "/"
str = ur'يوجد أشخاص يشترون العملة العراقي ويحتفظون بها إلى أن ترتفع سعر البيع، فهل هذا حلال أم حرام؟ (تجارة العملة حرام أم حلال)؟'
print st.tag_sents([str])


# english_tagger_path = 'D:/research/stanford-postagger-full-2015-01-30/models/english-bidirectional-distsim.tagger'
# english_tag = POSTagger(english_tagger_path, jar_path, encoding="utf-8")
# english_text = 'What is the airspeed of an unladen swallow ?'
# print english_tag.tag(english_text.split())
#
#
import jpype
import os.path
jarpath = os.path.join(os.path.abspath('.'), r"D:/research/stanford-postagger-full-2015-01-30/")
jpype.startJVM(jpype.getDefaultJVMPath(), "-Djava.ext.dirs=%s" % jarpath)

tagger_class = jpype.JClass("edu.stanford.nlp.tagger.maxent.MaxentTagger")
tagger = tagger_class(tagger_path)
res = tagger.tagString(str)
# res = tagger.tagTokenizedString(str)

jpype.java.lang.System.out.println(res)
# print(type(res))

print res

import os
import tempfile
from subprocess import PIPE
import warnings

from nltk.internals import find_file, find_jar, config_java, java, _java_options
from nltk.tag.api import TaggerI
from nltk import compat





_cmd =  ['edu.stanford.nlp.tagger.maxent.MaxentTagger',
                '-model', tagger_path, '-tokenize', 'false']
encoding = "utf-8"

def tag_sents(sentences):

    # Create a temporary input file
    _input_fh, _input_file_path = tempfile.mkstemp(text=True)

    _cmd.extend(['-encoding', encoding])
    _cmd.extend(['-textFile', _input_file_path])

    # Write the actual sentences to the temporary input file
    _input_fh = os.fdopen(_input_fh, 'wb')
    _input_fh.write(sentences.encode(encoding))
    _input_fh.close()

    # Run the tagger and get the output
    stanpos_output, _stderr = java(_cmd,classpath=jar_path,
                                                   stdout=PIPE, stderr=PIPE)
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
        tagged_sentences.append(sentence)
    return tagged_sentences

print "the result is "
print tag_sents(str)