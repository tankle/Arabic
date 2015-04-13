# -*- coding: utf-8 -*-
__author__ = 'Cong Tan'



import os

dir = ur"D:\百度云同步盘\Research\semeval2015\task3\arabic\semeval2015-task3-arabic-data\datasets"
trans_old_test_file_name = dir + os.sep + "trans_test_task3_Arabic.xml"
old_test_file_name = dir + os.sep + "test_task3_Arabic.xml"

"""
将官方的测试集结果更新到xml文件中
"""

basedir = ur"D:\百度云同步盘\Research\semeval2015\task3\SemEval2015_task3-submissions-last_only\_gold"
gold_txt = "QA.Arabic.test.gold"

#comment 的 cgold 标签
gold_dict = {}
with open(basedir+os.sep+gold_txt, "rb") as f:
    for line in f:
        line = line.split("\t")
        gold_dict[line[0].strip()] = line[1].strip()

from bs4 import BeautifulSoup as bs

def update_file(inname):
    print "running ", inname
    with open(inname, 'r') as f:
        content = f.read()
        data = bs(content, 'xml')
        root = data.root
        for ques in root.findChildren('Question'):
            qid = ques['QID']
            # print "processing , ", qid
            for ans in ques.findChildren('Answer'):
                cid = ans['CID']
                tmp_id = qid + "-" + cid
                print tmp_id
                if tmp_id in gold_dict:
                    ans['CGOLD'] = gold_dict[tmp_id]
                else:
                    print tmp_id, " is not in the gold_dict"
                    exit(0)
        outname = inname[:inname.rfind(".")] + "_new.xml"
        with open(outname, 'wb') as ff:
            ff.write(data.prettify(encoding="UTF-8"))

update_file(old_test_file_name)
update_file(trans_old_test_file_name)


